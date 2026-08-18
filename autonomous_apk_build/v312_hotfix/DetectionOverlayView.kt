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

/**
 * Single-layer live overlay.
 * Exactly ONE geometric rectangle is drawn per LiveDetection.
 * Labels are plain text with a small shadow: no filled panel and no second outline rectangle.
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        textSize = 28f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
        setShadowLayer(3.5f, 1.2f, 1.2f, Color.BLACK)
    }

    @Volatile private var items: List<LiveDetection> = emptyList()
    @Volatile private var frameWidth = 1
    @Volatile private var frameHeight = 1
    @Volatile private var mirrored = false

    fun submit(next: List<LiveDetection>, width: Int, height: Int, mirror: Boolean) {
        items = next
        frameWidth = max(1, width)
        frameHeight = max(1, height)
        mirrored = mirror
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val fw = frameWidth.toFloat()
        val fh = frameHeight.toFloat()
        val scale = min(width / fw, height / fh)
        val dx = (width - fw * scale) * .5f
        val dy = (height - fh * scale) * .5f

        for (item in items) {
            val d = item.detection
            val left = if (mirrored) dx + (fw - d.x2) * scale else dx + d.x1 * scale
            val right = if (mirrored) dx + (fw - d.x1) * scale else dx + d.x2 * scale
            val top = dy + d.y1 * scale
            val bottom = dy + d.y2 * scale
            if (right <= left || bottom <= top) continue

            val color = when (item.state) {
                DecisionState.HEALTHY -> Color.rgb(70, 220, 115)
                DecisionState.DISEASE -> Color.rgb(255, 82, 82)
                DecisionState.REVIEW -> Color.rgb(255, 193, 7)
            }
            boxPaint.color = color
            textPaint.color = color

            // The ONLY rectangle drawn for this diagnosis.
            canvas.drawRoundRect(RectF(left, top, right, bottom), 10f, 10f, boxPaint)

            val state = when (item.state) {
                DecisionState.HEALTHY -> "ΥΓΙΕΣ"
                DecisionState.DISEASE -> "ΑΣΘΕΝΕΙΑ"
                DecisionState.REVIEW -> "ΕΛΕΓΧΟΣ"
            }
            val text = "$state · ${DiseaseCatalog.label(item.displayClassId)} · ${(d.confidence * 100).toInt()}%"
            val x = (left + 6f).coerceIn(4f, (width - 8).toFloat())
            val y = (top + textPaint.textSize + 6f)
                .coerceIn(textPaint.textSize + 4f, (height - 4).toFloat())
            canvas.drawText(text, x, y, textPaint)
        }
    }
}
