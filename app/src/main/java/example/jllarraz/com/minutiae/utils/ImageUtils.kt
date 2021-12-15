package example.jllarraz.com.minutiae.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.renderscript.ScriptIntrinsicBlur

import android.renderscript.Allocation
import android.renderscript.Element

import android.renderscript.RenderScript




object ImageUtils {

    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    fun blur(image: Bitmap?, blurRadius:Float = 25f, context: Context,): Bitmap? {
        if (null == image) return null
        val outputBitmap = Bitmap.createBitmap(image)

        val renderScript = RenderScript.create(context)
        val tmpIn = Allocation.createFromBitmap(renderScript, image)
        val tmpOut = Allocation.createFromBitmap(renderScript, outputBitmap)

        //Intrinsic Gausian blur filter
        val theIntrinsic = ScriptIntrinsicBlur.create(renderScript, Element.U8_4(renderScript))
        theIntrinsic.setRadius(blurRadius)
        theIntrinsic.setInput(tmpIn)
        theIntrinsic.forEach(tmpOut)
        tmpOut.copyTo(outputBitmap)
        return outputBitmap
    }
}