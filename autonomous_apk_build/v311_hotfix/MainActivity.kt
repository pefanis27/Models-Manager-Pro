package com.autonomousdiagnosis.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    companion object {
        private const val CAMERA_PERMISSION_REQUEST=401
        private const val DETECTION_CONFIDENCE=0.25f
        private const val IOU_THRESHOLD=0.45f
        private const val ANALYSIS_INTERVAL_MS=100L
    }

    private lateinit var preview: PreviewView
    private lateinit var overlay: DetectionOverlayView
    private lateinit var status: TextView
    private lateinit var details: TextView
    private lateinit var gpuSwitch: Switch
    private lateinit var switchCamera: Button

    private val inferenceExecutor=Executors.newSingleThreadExecutor()
    private val modelReady=AtomicBoolean(false)
    private val tracker=VideoDetectionTracker()
    private var provider: ProcessCameraProvider?=null
    private var lensFacing=androidx.camera.core.CameraSelector.LENS_FACING_BACK
    private var lastAnalysisAt=0L
    private var lastCompletedAt=0L
    private var fpsEma=0.0
    private var engineLabel="NCNN"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window,false)
        buildUi()
        gpuSwitch.setOnCheckedChangeListener { _, checked ->
            if (!gpuSwitch.isEnabled) return@setOnCheckedChangeListener
            reloadModel(checked)
        }
        switchCamera.setOnClickListener {
            lensFacing=if(lensFacing==androidx.camera.core.CameraSelector.LENS_FACING_BACK)
                androidx.camera.core.CameraSelector.LENS_FACING_FRONT else androidx.camera.core.CameraSelector.LENS_FACING_BACK
            tracker.clear(); overlay.submit(emptyList(),1,1,false); bindCamera()
        }
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED) initialize()
        else ActivityCompat.requestPermissions(this,arrayOf(Manifest.permission.CAMERA),CAMERA_PERMISSION_REQUEST)
    }

    private fun buildUi() {
        val root=FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        preview=PreviewView(this).apply {
            scaleType=PreviewView.ScaleType.FIT_CENTER
            implementationMode=PreviewView.ImplementationMode.PERFORMANCE
        }
        overlay=DetectionOverlayView(this)
        root.addView(preview,FrameLayout.LayoutParams(-1,-1))
        root.addView(overlay,FrameLayout.LayoutParams(-1,-1))

        val top=LinearLayout(this).apply {
            orientation=LinearLayout.VERTICAL
            setPadding(dp(14),dp(10),dp(14),dp(10))
            setBackgroundColor(Color.argb(175,8,45,23))
        }
        status=TextView(this).apply {
            text="Autonomous Diagnosis · φόρτωση…"; setTextColor(Color.WHITE); textSize=18f
            setTypeface(typeface,android.graphics.Typeface.BOLD)
        }
        details=TextView(this).apply {
            text="YOLO26s · 640×640 · NCNN"; setTextColor(Color.rgb(205,235,212)); textSize=13f
        }
        top.addView(status); top.addView(details)
        root.addView(top,FrameLayout.LayoutParams(-1,-2).apply { gravity=Gravity.TOP })

        val bottom=LinearLayout(this).apply {
            orientation=LinearLayout.VERTICAL; setPadding(dp(12),dp(8),dp(12),dp(8)); setBackgroundColor(Color.argb(175,8,45,23))
        }
        gpuSwitch=Switch(this).apply {
            text="Vulkan GPU"; setTextColor(Color.WHITE); isChecked=true; isEnabled=false; minimumHeight=dp(48)
        }
        switchCamera=Button(this).apply {
            text="Αλλαγή κάμερας"; isAllCaps=false; minimumHeight=dp(52); maxLines=2
        }
        bottom.addView(gpuSwitch,LinearLayout.LayoutParams(-1,-2))
        bottom.addView(switchCamera,LinearLayout.LayoutParams(-1,-2).apply { topMargin=dp(6) })
        root.addView(bottom,FrameLayout.LayoutParams(-1,-2).apply { gravity=Gravity.BOTTOM })

        ViewCompat.setOnApplyWindowInsetsListener(root) { _,insets ->
            val safe=insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            top.setPadding(dp(14)+safe.left,dp(10)+safe.top,dp(14)+safe.right,dp(10))
            bottom.setPadding(dp(12)+safe.left,dp(8),dp(12)+safe.right,dp(8)+safe.bottom)
            insets
        }
        setContentView(root); ViewCompat.requestApplyInsets(root)
    }

    private fun initialize() {
        inferenceExecutor.execute {
            val hasGpu=try { NativeDetector.nativeHasGpu() } catch(_:Throwable){ false }
            var ok=false
            var vulkan=false
            if(hasGpu) {
                ok=try { NativeDetector.nativeInit(assets,true) } catch(_:Throwable){ false }
                vulkan=ok
            }
            if(!ok) ok=try { NativeDetector.nativeInit(assets,false) } catch(_:Throwable){ false }
            modelReady.set(ok); engineLabel=if(vulkan) "NCNN · Vulkan" else "NCNN · CPU"
            runOnUiThread {
                gpuSwitch.isEnabled=false
                gpuSwitch.isChecked=ok && vulkan
                gpuSwitch.isEnabled=ok && hasGpu
                status.text=if(ok) "LIVE · έτοιμο" else "ΜΟΝΤΕΛΟ ΔΕΝ ΦΟΡΤΩΘΗΚΕ"
                details.text=if(ok) "YOLO26s · 640×640 · $engineLabel" else "Αποτυχία φόρτωσης νέου μοντέλου"
                bindCamera()
            }
        }
    }

    private fun reloadModel(requestVulkan:Boolean) {
        modelReady.set(false)
        status.text="Επαναφόρτωση μοντέλου…"
        inferenceExecutor.execute {
            val hasGpu=try { NativeDetector.nativeHasGpu() } catch(_:Throwable){ false }
            var vulkan=requestVulkan && hasGpu
            var ok=try {
                NativeDetector.nativeClose()
                NativeDetector.nativeInit(assets,vulkan)
            } catch(_:Throwable){ false }
            if(!ok && vulkan) {
                vulkan=false
                ok=try { NativeDetector.nativeInit(assets,false) } catch(_:Throwable){ false }
            }
            tracker.clear()
            engineLabel=if(ok && vulkan) "NCNN · Vulkan" else "NCNN · CPU"
            modelReady.set(ok)
            runOnUiThread {
                overlay.submit(emptyList(),1,1,false)
                gpuSwitch.isEnabled=false
                gpuSwitch.isChecked=ok && vulkan
                gpuSwitch.isEnabled=ok && hasGpu
                status.text=if(ok) "LIVE · έτοιμο" else "Αποτυχία φόρτωσης μοντέλου"
                details.text=if(ok) "YOLO26s · 640×640 · $engineLabel" else "NCNN model reload failed"
            }
        }
    }

    private fun bindCamera() {
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED) return
        val future=ProcessCameraProvider.getInstance(this)
        future.addListener({
            val p=future.get(); provider=p; p.unbindAll()
            val resolution=ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY).build()
            val cameraPreview=Preview.Builder().setResolutionSelector(resolution).build().also { it.surfaceProvider=preview.surfaceProvider }
            val analysis=ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setOutputImageRotationEnabled(true)
                .build()
            analysis.setAnalyzer(inferenceExecutor) { image ->
                try {
                    val now=SystemClock.elapsedRealtime()
                    if(!modelReady.get() || now-lastAnalysisAt<ANALYSIS_INTERVAL_MS) return@setAnalyzer
                    lastAnalysisAt=now
                    val plane=image.planes.firstOrNull() ?: return@setAnalyzer
                    val source=plane.buffer; source.rewind()
                    val buffer:ByteBuffer=if(source.isDirect) source else ByteBuffer.allocateDirect(source.remaining()).also { it.put(source); it.rewind() }
                    val started=SystemClock.elapsedRealtimeNanos()
                    val raw=NativeDetector.nativeDetect(buffer,image.width,image.height,plane.rowStride,DETECTION_CONFIDENCE,IOU_THRESHOLD)
                    val inferenceMs=(SystemClock.elapsedRealtimeNanos()-started)/1_000_000.0
                    val stable=tracker.update(NativeDetector.decode(raw),now)
                    updateFps(now)
                    val w=image.width; val h=image.height; val mirror=lensFacing==androidx.camera.core.CameraSelector.LENS_FACING_FRONT
                    runOnUiThread { render(stable,w,h,mirror,inferenceMs) }
                } catch(t:Throwable) {
                    runOnUiThread { status.text="Σφάλμα inference"; details.text=t.message ?: t.javaClass.simpleName }
                } finally { image.close() }
            }
            val selector=androidx.camera.core.CameraSelector.Builder().requireLensFacing(lensFacing).build()
            try { p.bindToLifecycle(this,selector,cameraPreview,analysis) }
            catch(_:Throwable) {
                if(lensFacing!=androidx.camera.core.CameraSelector.LENS_FACING_BACK) { lensFacing=androidx.camera.core.CameraSelector.LENS_FACING_BACK; bindCamera() }
            }
        },ContextCompat.getMainExecutor(this))
    }

    private fun render(items:List<LiveDetection>,w:Int,h:Int,mirror:Boolean,inferenceMs:Double) {
        overlay.submit(items,w,h,mirror)
        val disease=items.count{it.state==DecisionState.DISEASE}; val healthy=items.count{it.state==DecisionState.HEALTHY}; val review=items.count{it.state==DecisionState.REVIEW}
        status.text=when {
            disease>0 -> "LIVE · ΑΣΘΕΝΕΙΑ $disease · ΥΓΙΗ $healthy"
            healthy>0 && review==0 -> "LIVE · ΥΓΙΕΣ · $healthy στόχοι"
            review>0 -> "LIVE · ΕΛΕΓΧΟΣ $review"
            else -> "LIVE · αναζήτηση φύλλων/σταφυλιών"
        }
        details.text="YOLO26s 640 · $engineLabel · ${"%.1f".format(fpsEma)} infer/s · ${inferenceMs.roundToInt()} ms · det≥25% · confirm≥65%"
    }

    private fun updateFps(now:Long) {
        if(lastCompletedAt>0) {
            val current=1000.0/(now-lastCompletedAt).coerceAtLeast(1L)
            fpsEma=if(fpsEma==0.0) current else fpsEma*.85+current*.15
        }
        lastCompletedAt=now
    }

    override fun onRequestPermissionsResult(requestCode:Int,permissions:Array<out String>,grantResults:IntArray) {
        super.onRequestPermissionsResult(requestCode,permissions,grantResults)
        if(requestCode==CAMERA_PERMISSION_REQUEST && grantResults.firstOrNull()==PackageManager.PERMISSION_GRANTED) initialize()
        else { status.text="Απαιτείται άδεια κάμερας"; details.text="Η live διάγνωση χρειάζεται CAMERA permission." }
    }

    override fun onDestroy() {
        provider?.unbindAll(); modelReady.set(false)
        inferenceExecutor.execute { try { NativeDetector.nativeClose() } catch(_:Throwable){} }
        inferenceExecutor.shutdown(); super.onDestroy()
    }
    private fun dp(v:Int)= (v*resources.displayMetrics.density).roundToInt()
}
