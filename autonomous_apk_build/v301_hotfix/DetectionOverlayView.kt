package com.autonomousdiagnosis.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
        textSize = 30f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val labelOutlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = Color.BLACK
        textSize = 30f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    @Volatile private var items: List<LiveDetection> = emptyList()
    @Volatile private var frameWidth: Int = 1
    @Volatile private var frameHeight: Int = 1
    @Volatile private var mirrorHorizontally: Boolean = false

    fun submit(detections: List<LiveDetection>, width: Int, height: Int, mirrored: Boolean) {
        items = detections
        frameWidth = max(1, width)
        frameHeight = max(1, height)
        mirrorHorizontally = mirrored
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val fw = frameWidth.toFloat()
        val fh = frameHeight.toFloat()
        val scale = min(width / fw, height / fh)
        val dx = (width - fw * scale) * 0.5f
        val dy = (height - fh * scale) * 0.5f

        for (item in items) {
            val d = item.detection
            val left = if (mirrorHorizontally) dx + (fw - d.x2) * scale else dx + d.x1 * scale
            val top = dy + d.y1 * scale
            val right = if (mirrorHorizontally) dx + (fw - d.x1) * scale else dx + d.x2 * scale
            val bottom = dy + d.y2 * scale
            if (right <= left || bottom <= top) continue

            boxPaint.color = when (item.state) {
                DecisionState.HEALTHY -> Color.rgb(70, 220, 115)
                DecisionState.DISEASE -> Color.rgb(255, 82, 82)
                DecisionState.REVIEW -> Color.rgb(255, 193, 7)
            }
            canvas.drawRoundRect(RectF(left, top, right, bottom), 12f, 12f, boxPaint)

            val stateText = when (item.state) {
                DecisionState.HEALTHY -> "ΥΓΙΕΣ"
                DecisionState.DISEASE -> "ΑΣΘΕΝΕΙΑ"
                DecisionState.REVIEW -> "ΕΠΑΝΕΞΕΤΑΣΗ"
            }
            val text = "$stateText · ${DiseaseCatalog.label(item.displayClassId)} · ${(d.confidence * 100).toInt()}%"
            val textX = left.coerceIn(4f, width.toFloat() - 4f)
            val baseline = (top - 10f).coerceAtLeast(labelPaint.textSize + 4f)
            labelPaint.color = boxPaint.color
            canvas.drawText(text, textX, baseline, labelOutlinePaint)
            canvas.drawText(text, textX, baseline, labelPaint)
        }
    }
}
