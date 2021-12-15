package example.jllarraz.com.minutiae.ui.views

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import example.jllarraz.com.minutiae.ui.views.CameraOverlayView

class CameraOverlayView : View {
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val innerPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val overlayRect = RectF()

    constructor(context: Context?) : super(context) {
        initPaints()
    }

    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs) {
        initPaints()
    }

    constructor(context: Context?, attrs: AttributeSet?, defStyle: Int) : super(
        context,
        attrs,
        defStyle
    ) {
        initPaints()
    }

    public override fun onDraw(canvas: Canvas) {
        overlayRect[width * PADDING, height * PADDING, width * (1.0f - PADDING)] =
            height * (1.0f - PADDING)
        canvas.drawOval(overlayRect, innerPaint)
        canvas.drawOval(overlayRect, borderPaint)
    }

    private fun initPaints() {
        borderPaint.color = Color.WHITE
        borderPaint.style = Paint.Style.STROKE
        borderPaint.strokeWidth = 6f
        borderPaint.setShadowLayer(12f, 0f, 0f, Color.GREEN)
        innerPaint.setARGB(0, 0, 0, 0)
        innerPaint.style = Paint.Style.FILL
        setLayerType(LAYER_TYPE_SOFTWARE, borderPaint)
    }

    companion object {
        const val PADDING = 0.25f
    }
}