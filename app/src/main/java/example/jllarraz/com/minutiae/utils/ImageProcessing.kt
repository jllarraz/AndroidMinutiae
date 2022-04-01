package example.jllarraz.com.minutiae.utils

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import org.opencv.imgproc.Imgproc
import kotlin.Throws
import org.opencv.features2d.Features2d
import example.jllarraz.com.minutiae.ui.views.CameraOverlayView
import android.graphics.BitmapFactory
import android.util.Log
import io.reactivex.Single
import org.opencv.BuildConfig
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import java.lang.Exception
import java.lang.IndexOutOfBoundsException
import java.lang.RuntimeException
import java.util.*

internal class ImageProcessing{
    private var paddingSize = 0

    private enum class Step{
        START_FINGERPRINT_EXTRACTION,
        SKIN_DETECTION,
        HISTOGRAM_EQUALIZATION,
        CLAHE_EQUALIZATION,
        ADAPTATIVE_THRESHOLD,
        BILATERAL_FILTER,
        FINGERPRINT_SKELETIZATION,
        RIDGE_THINNING,
        MINUTIAE_EXTRACTION,
        FINISH_FINGERPRINT_EXTRACTION,
    }


    // convert to float, very important
    /*
    * get fingerprint skeleton image. Large part of this code is copied from
    * https://github.com/noureldien/FingerprintRecognition/blob/master/Java/src/com/fingerprintrecognition/ProcessActivity.java
    */
    fun processedImage(data: ByteArray, context: Context?, isClahe:Boolean=false, isAdaptativeThreshold:Boolean=false, isBilateralFilter:Boolean=false): Single<Bitmap>{
        return Single.fromCallable {
            onNextStep(Step.START_FINGERPRINT_EXTRACTION, context)
            var imageColor = bytesToMat(data)
            imageColor = rotateImage(imageColor)
            imageColor = cropFingerprint(imageColor)

            val lower = Scalar(0.toDouble(), 20.toDouble(), 20.toDouble())
            val upper = Scalar(20.toDouble(), 255.toDouble(), 255.toDouble())

            imageColor = skinDetection(imageColor, lower, upper)

            onNextStep(Step.SKIN_DETECTION, context)

            val image = Mat(imageColor.rows(), imageColor.cols(), CvType.CV_8UC1)
            Imgproc.cvtColor(imageColor, image, Imgproc.COLOR_BGR2GRAY)
            val rows = image.rows()
            val cols = image.cols()

            // apply histogram equalization
            var equalizedHistogram = Mat(rows, cols, CvType.CV_32FC1)
            if(!isClahe) {
                Imgproc.equalizeHist(image, equalizedHistogram)
                onNextStep(Step.HISTOGRAM_EQUALIZATION, context)
            }else{
                val clahe = Imgproc.createCLAHE()
                clahe.clipLimit = 2.0
                clahe.tilesGridSize = Size(4.toDouble(),4.toDouble())
                clahe.apply(image, equalizedHistogram)
                onNextStep(Step.CLAHE_EQUALIZATION, context)
            }

            if(isAdaptativeThreshold){
                val adaptativeThreshold = Mat(rows, cols, CvType.CV_32FC1)
                Imgproc.adaptiveThreshold(equalizedHistogram, adaptativeThreshold, 255.toDouble(), Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.toDouble())
                equalizedHistogram =adaptativeThreshold
                onNextStep(Step.ADAPTATIVE_THRESHOLD, context)
            }

            if(isBilateralFilter) {
                val bilateralFilter = Mat(rows, cols, CvType.CV_32FC1)
                Imgproc.bilateralFilter(
                    equalizedHistogram,
                    bilateralFilter,
                    9,
                    20.toDouble(),
                    20.toDouble()
                )
                equalizedHistogram = bilateralFilter
                onNextStep(Step.BILATERAL_FILTER, context)
            }

            // convert to float, very important
            val floatedClaheAdaptative = Mat(rows, cols, CvType.CV_32FC1)
            equalizedHistogram.convertTo(floatedClaheAdaptative, CvType.CV_32FC1)
            var skeletonClaheAdaptative = getSkeletonImage(
                src = floatedClaheAdaptative,
                rows = rows,
                cols = cols,

            )


            onNextStep(Step.FINGERPRINT_SKELETIZATION, context)
            skeletonClaheAdaptative = thinning(skeletonClaheAdaptative)
            onNextStep(Step.RIDGE_THINNING, context)
            val skeleton_with_keypoints = detectMinutiae(skeletonClaheAdaptative, 1)
            onNextStep(Step.MINUTIAE_EXTRACTION, context)
            val mat2Bitmap = mat2Bitmap(skeleton_with_keypoints, Imgproc.COLOR_RGB2RGBA)
            onNextStep(Step.FINISH_FINGERPRINT_EXTRACTION, context)
            mat2Bitmap
        }

    }

    private fun onNextStep(step: Step, context: Context?){
        var intentRequest: Intent?=null
        when(step){
            Step.START_FINGERPRINT_EXTRACTION ->{
                intentRequest = Intent(ACTION_STEP_START_FINGERPRINT_EXTRACTION)
            }
            Step.SKIN_DETECTION ->{
                intentRequest = Intent(ACTION_STEP_SKIN_DETECTION)
            }
            Step.HISTOGRAM_EQUALIZATION ->{
                intentRequest = Intent(ACTION_STEP_HISTOGRAM_EQUALIZATION)
            }
            Step.CLAHE_EQUALIZATION ->{
                intentRequest = Intent(ACTION_STEP_CLAHE_EQUALIZATION)
            }
            Step.ADAPTATIVE_THRESHOLD ->{
                intentRequest = Intent(ACTION_STEP_ADAPTATIVE_THRESHOLD)
            }
            Step.BILATERAL_FILTER ->{
                intentRequest = Intent(ACTION_STEP_BILATERAL_FILTER)
            }
            Step.FINGERPRINT_SKELETIZATION ->{
                intentRequest = Intent(ACTION_STEP_FINGERPRINT_SKELETIZATION)
            }
            Step.RIDGE_THINNING ->{
                intentRequest = Intent(ACTION_STEP_RIDGE_THINNING)
            }
            Step.MINUTIAE_EXTRACTION ->{
                intentRequest = Intent(ACTION_STEP_MINUTIAE_EXTRATION)
            }
            Step.FINISH_FINGERPRINT_EXTRACTION ->{
                intentRequest = Intent(ACTION_STEP_FINISH_FINGERPRINT_EXTRACTION)
            }
        }
        intentRequest?.let {
            context?.sendBroadcast(it)
        }
    }


    private fun neighbourCount(skeleton: Mat, row: Int, col: Int): Int {
        var cn = 0
        if (skeleton[row - 1, col - 1][0] != 0.toDouble()) cn++
        if (skeleton[row - 1, col][0] != 0.toDouble()) cn++
        if (skeleton[row - 1, col + 1][0] != 0.toDouble()) cn++
        if (skeleton[row, col - 1][0] != 0.toDouble()) cn++
        if (skeleton[row, col + 1][0] != 0.toDouble()) cn++
        if (skeleton[row + 1, col - 1][0] != 0.toDouble()) cn++
        if (skeleton[row + 1, col][0] != 0.toDouble()) cn++
        if (skeleton[row + 1, col + 1][0] != 0.toDouble()) cn++
        return cn
    }

    private fun detectMinutiae(skeleton: Mat, border: Int): Mat {
        val minutiaeSet = HashSet<Minutiae>()
        println("Detecting minutiae")
        for (c in border until skeleton.cols() - border) {
            for (r in border until skeleton.rows() - border) {
                val point = skeleton[r, c][0]
                if (point != 0.0) {  // Not black
                    val cn = neighbourCount(skeleton, r, c)
                    if (cn == 1) minutiaeSet.add(
                        Minutiae(
                            c,
                            r,
                            Minutiae.Type.RIDGEENDING
                        )
                    ) else if (cn == 3) minutiaeSet.add(Minutiae(c, r, Minutiae.Type.BIFURCATION))
                }
            }
        }
        println("filtering minutiae")
        val filteredMinutiae = filterMinutiae(minutiaeSet, skeleton)
        println("number of minutiae: " + filteredMinutiae.size)
        val result = Mat()
        println("Drawing minutiae")
        Imgproc.cvtColor(skeleton, result, Imgproc.COLOR_GRAY2RGB)
        val red = doubleArrayOf(255.0, 0.0, 0.0)
        val green = doubleArrayOf(0.0, 255.0, 0.0)
        for (m in filteredMinutiae) {
            var color: DoubleArray
            color = if (m.type == Minutiae.Type.BIFURCATION) green else red
            result.put(m.y, m.x, *color)
            result.put(m.y, m.x - 1, *color)
            result.put(m.y, m.x + 1, *color)
            result.put(m.y - 1, m.x, *color)
            result.put(m.y + 1, m.x, *color)
        }
        val keypoints = MatOfKeyPoint()
        keypoints.fromArray(*minutiaeToKeyPoints(skeleton, filteredMinutiae))
        Companion.keypoints = keypoints
        val extractor = ORB.create()
        //DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        val descriptors = Mat()
        extractor.compute(skeleton, keypoints, descriptors)
        Companion.descriptors = descriptors
        return result
    }

    private fun minutiaeToKeyPoints(skeleton: Mat, minutiae: HashSet<Minutiae>): Array<KeyPoint?> {
        val result = arrayOfNulls<KeyPoint>(minutiae.size)
        var index = 0
        val size = 1f
        for (m in minutiae) {
            var k: KeyPoint
            var angle = -1f
            val response = 1f
            val octave = 1
            var class_id: Int
            if (m.type == Minutiae.Type.RIDGEENDING) {
                angle = getMinutiaeAngle(skeleton, m)
                class_id = Minutiae.RIDGE_ENDING_LABEL
            } else {
                class_id = Minutiae.BIFURCATION_LABEL
            }
            result[index] =
                KeyPoint(m.x.toFloat(), m.y.toFloat(), size, angle, response, octave, class_id)
            index++
        }
        return result
    }

    private fun getMinutiaeAngle(skeleton: Mat, m: Minutiae): Float {
        val direction: Minutiae
        direction = try {
            followRidge(skeleton, 5, m.x, m.y, m.x, m.y)
        } catch (e: Exception) {
            e.printStackTrace()
            return (-1).toFloat()
        }
        val length = Math.sqrt(
            Math.pow(
                (direction.x - m.x).toDouble(),
                2.0
            ) + Math.pow((direction.y - m.y).toDouble(), 2.0)
        )
        val cosine = (direction.y - m.y) / length
        var angle = Math.acos(cosine).toFloat()
        if (direction.x - m.x < 0) angle = -angle + 2 * Math.PI.toFloat()
        return angle
    }

    @Throws(Exception::class)
    private fun followRidge(
        skeleton: Mat,
        length: Int,
        currentX: Int,
        currentY: Int,
        previousX: Int,
        previousY: Int
    ): Minutiae {
        if (length == 0) return Minutiae(currentX, currentY, Minutiae.Type.RIDGEENDING)
        if (currentY >= skeleton.rows() - 1 || currentY == 0 || currentX >= skeleton.cols() - 1 || currentX == 0) throw Exception(
            "out of bounds"
        )
        var _x: Int
        var _y: Int
        _x = currentX - 1
        _y = currentY - 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)) return followRidge(
            skeleton,
            length - 1,
            _x,
            _y,
            currentX,
            currentY
        )
        _x = currentX - 1
        _y = currentY
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)) return followRidge(
            skeleton,
            length - 1,
            _x,
            _y,
            currentX,
            currentY
        )
        _x = currentX - 1
        _y = currentY + 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)) return followRidge(
            skeleton,
            length - 1,
            _x,
            _y,
            currentX,
            currentY
        )
        _x = currentX
        _y = currentY - 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)) return followRidge(
            skeleton,
            length - 1,
            _x,
            _y,
            currentX,
            currentY
        )
        _x = currentX
        _y = currentY + 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)) return followRidge(
            skeleton,
            length - 1,
            _x,
            _y,
            currentX,
            currentY
        )
        _x = currentX + 1
        _y = currentY - 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)) return followRidge(
            skeleton,
            length - 1,
            _x,
            _y,
            currentX,
            currentY
        )
        _x = currentX + 1
        _y = currentY
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)) return followRidge(
            skeleton,
            length - 1,
            _x,
            _y,
            currentX,
            currentY
        )
        _x = currentX + 1
        _y = currentY + 1
        return followRidge(skeleton, length - 1, _x, _y, currentX, currentY)
    }

    private fun followRidgeCheck(
        skeleton: Mat,
        x: Int,
        y: Int,
        previousX: Int,
        previousY: Int
    ): Boolean {
        return if (x == previousX && y == previousY) false else skeleton[y, x][0] != 0.toDouble()
    }

    private fun removeMinutiae(
        minDistance: Int,
        source: HashSet<Minutiae>,
        target: HashSet<Minutiae>
    ) {
        val toBeRemoved = HashSet<Minutiae>()
        val check = HashSet(source)
        for (m in source) {
            if (toBeRemoved.contains(m)) continue
            var ok = true
            check.remove(m)
            for (m2 in check) {
                if (m.euclideanDistance(m2) < minDistance) {
                    ok = false
                    toBeRemoved.add(m2)
                }
            }
            if (ok) target.add(m) else toBeRemoved.add(m)
        }
    }

    private fun filterMinutiae(src: HashSet<Minutiae>, skeleton: Mat): HashSet<Minutiae> {
        val mask = snapShotMask(skeleton.rows(), skeleton.cols(), paddingSize + 5)
        val ridgeEnding = HashSet<Minutiae>()
        val bifurcation = HashSet<Minutiae>()
        val filtered = HashSet<Minutiae>()
        for (m in src) {
            if (mask[m.y, m.x][0] > 0) {  // filter out borders
                if (m.type == Minutiae.Type.BIFURCATION) ridgeEnding.add(m) else bifurcation.add(m)
            }
        }
        val minDistance = 5
        removeMinutiae(minDistance, ridgeEnding, filtered)
        removeMinutiae(minDistance, bifurcation, filtered)
        return filtered
    }

    private fun detectFeatures(skeleton: Mat, edges: Mat): Mat {
        val keypoints = MatOfKeyPoint()
        val star = ORB.create()
        val brief = ORB.create()

        //FeatureDetector star = FeatureDetector.create(FeatureDetector.ORB);
        //DescriptorExtractor brief = DescriptorExtractor.create(DescriptorExtractor.ORB);
        star.detect(skeleton, keypoints)
        Companion.keypoints = keypoints
        val keypointArray = keypoints.toArray()
        val filteredKeypointArray = ArrayList<KeyPoint>(keypointArray.size)
        var filterCount = 0
        for (k in keypointArray) {
            if (edges[k.pt.y.toInt(), k.pt.x.toInt()][0] <= 0.0) {
                k.size /= 8f
                filteredKeypointArray.add(k)
            } else {
                filterCount++
            }
        }
        Log.d(TAG, String.format("Filtered %s Keypoints", filterCount))
        keypoints.fromList(filteredKeypointArray)
        val descriptors = Mat()
        brief.compute(skeleton, keypoints, descriptors)
        Companion.descriptors = descriptors
        val results = Mat()
        val color = Scalar(255.toDouble(), 0.toDouble(), 0.toDouble()) // RGB
        Features2d.drawKeypoints(
            skeleton,
            keypoints,
            results,
            color,
            Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        return results
    }

    /**
     * define the upper and lower boundaries of the HSV pixel
     * intensities to be considered 'skin'
     */
    fun skinDetection(src: Mat,
                      lower:Scalar = Scalar(0.toDouble(), 48.toDouble(), 80.toDouble()),
                      upper:Scalar = Scalar(20.toDouble(), 255.toDouble(), 255.toDouble())
    ): Mat {
        // Convert to HSV
        val hsvFrame = Mat(src.rows(), src.cols(), CvType.CV_8U, Scalar(3.toDouble()))
        Imgproc.cvtColor(src, hsvFrame, Imgproc.COLOR_RGB2HSV, 3)

        // Mask the image for skin colors
        val skinMask = Mat(hsvFrame.rows(), hsvFrame.cols(), CvType.CV_8U, Scalar(3.toDouble()))
        Core.inRange(hsvFrame, lower, upper, skinMask)
        //        currentSkinMask = new Mat(hsvFrame.rows(), hsvFrame.cols(), CvType.CV_8U, new Scalar(3));
//        skinMask.copyTo(currentSkinMask);

        // apply a series of erosions and dilations to the mask
        // using an elliptical kernel
        val kernelSize = Size(11.toDouble(), 11.toDouble())
        val anchor = Point(-1.toDouble(), -1.toDouble())
        val iterations = 2
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, kernelSize)
        Imgproc.erode(skinMask, skinMask, kernel, anchor, iterations)
        Imgproc.dilate(skinMask, skinMask, kernel, anchor, iterations)

        // blur the mask to help remove noise, then apply the
        // mask to the frame
        val ksize = Size(3.toDouble(), 3.toDouble())
        val skin = Mat(skinMask.rows(), skinMask.cols(), CvType.CV_8U, Scalar(3.toDouble()))
        Imgproc.GaussianBlur(skinMask, skinMask, ksize, 0.0)
        Core.bitwise_and(src, src, skin, skinMask)
        return skin
    }

    private fun getSkeletonImage(src: Mat,
                                 rows: Int,
                                 cols: Int,
                                 ridgeSegmentPaddingBlockSize:Int = 24,
                                 ridgeSegmentPaddingThreshold:Double = 0.05,
                                 ridgeOrientationGradientSigma:Double = 1.0,
                                 ridgeOrientationBlockSigma:Double = 13.0,
                                 ridgeOrientationOrientSmoothSigma:Double = 15.0,
                                 ridgeFrequencyBlockSize:Int = 36,
                                 ridgeFrequencyWindowSize:Int = 5,
                                 ridgeFrequencyMinWaveLength:Int = 5,
                                 ridgeFrequencyMaxWaveLength:Int = 25,
                                 ridgeFilterFilterSize:Double = 1.9
    ): Mat {
        // step 1: get ridge segment by padding then do block process
        val blockSize = ridgeSegmentPaddingBlockSize
        val threshold = ridgeSegmentPaddingThreshold
        val padded = imagePadding(src, blockSize)
        val imgRows = padded.rows()
        val imgCols = padded.cols()
        val matRidgeSegment = Mat(imgRows, imgCols, CvType.CV_32FC1)
        val segmentMask = Mat(imgRows, imgCols, CvType.CV_8UC1)
        ridgeSegment(padded, matRidgeSegment, segmentMask, blockSize, threshold)

        // step 2: get ridge orientation
        val gradientSigma = ridgeOrientationGradientSigma
        val blockSigma = ridgeOrientationBlockSigma
        val orientSmoothSigma = ridgeOrientationOrientSmoothSigma
        val matRidgeOrientation = Mat(imgRows, imgCols, CvType.CV_32FC1)
        ridgeOrientation(
            matRidgeSegment,
            matRidgeOrientation,
            gradientSigma,
            blockSigma,
            orientSmoothSigma
        )

        // step 3: get ridge frequency
        val fBlockSize = ridgeFrequencyBlockSize
        val fWindowSize = ridgeFrequencyWindowSize
        val fMinWaveLength = ridgeFrequencyMinWaveLength
        val fMaxWaveLength = ridgeFrequencyMaxWaveLength
        val matFrequency = Mat(imgRows, imgCols, CvType.CV_32FC1)
        val medianFreq = ridgeFrequency(
            matRidgeSegment,
            segmentMask,
            matRidgeOrientation,
            matFrequency,
            fBlockSize,
            fWindowSize,
            fMinWaveLength,
            fMaxWaveLength
        )

        // step 4: get ridge filter
        val matRidgeFilter = Mat(imgRows, imgCols, CvType.CV_32FC1)
        val filterSize = ridgeFilterFilterSize
        val padding = ridgeFilter(
            matRidgeSegment,
            matRidgeOrientation,
            matFrequency,
            matRidgeFilter,
            filterSize,
            filterSize,
            medianFreq
        )
        paddingSize = padding

        // step 5: enhance image after ridge filter
        val matEnhanced = Mat(imgRows, imgCols, CvType.CV_8UC1)
        enhancement(matRidgeFilter, matEnhanced, blockSize, rows, cols, padding)
        return matEnhanced
    }

    private fun mat2Bitmap(src: Mat, code: Int): Bitmap {
        val rgbaMat = Mat(src.width(), src.height(), CvType.CV_8UC4)
        Imgproc.cvtColor(src, rgbaMat, code, 4)
        val bmp = Bitmap.createBitmap(rgbaMat.cols(), rgbaMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbaMat, bmp)
        return bmp
    }

    private fun cropFingerprint(src: Mat): Mat {
        val rowStart = (CameraOverlayView.PADDING * src.rows()).toInt()
        val rowEnd = ((1 - CameraOverlayView.PADDING) * src.rows()).toInt()
        val colStart = (CameraOverlayView.PADDING * src.cols()).toInt()
        val colEnd = ((1 - CameraOverlayView.PADDING) * src.cols()).toInt()
        val rowRange = Range(rowStart, rowEnd)
        val colRange = Range(colStart, colEnd)
        return src.submat(rowRange, colRange)
    }

    private fun bytesToMat(data: ByteArray): Mat {
        // Scale down the image for performance
        var bmp = BitmapFactory.decodeByteArray(data, 0, data.size)
        val targetWidth = 1200
        if (bmp.width > targetWidth) {
            val scaleDownFactor = targetWidth.toFloat() / bmp.width
            bmp = Bitmap.createScaledBitmap(
                bmp,
                (bmp.width * scaleDownFactor).toInt(),
                (bmp.height * scaleDownFactor).toInt(),
                true
            )
        }
        val BGRImage = Mat(bmp.width, bmp.height, CvType.CV_8UC3)
        Utils.bitmapToMat(bmp, BGRImage)
        return BGRImage
    }

    private fun emptyMat(width: Int, height: Int, dimension: Int): Mat {
        return Mat(width, height, CvType.CV_8U, Scalar(dimension.toDouble()))
    }

    /**
     * OpenCV only supports landscape pictures, so we gotta rotate 90 degrees.
     */
    private fun rotateImage(image: Mat): Mat {
        val result = emptyMat(image.rows(), image.cols(), 3)
        Core.transpose(image, result)
        Core.flip(result, result, 1)
        return result
    }
    // region FingerprintRecognition
    /**
     * Apply padding to the image.
     */
    private fun imagePadding(source: Mat, blockSize: Int): Mat {
        val width = source.width()
        val height = source.height()
        var bottomPadding = 0
        var rightPadding = 0
        if (width % blockSize != 0) {
            bottomPadding = blockSize - width % blockSize
        }
        if (height % blockSize != 0) {
            rightPadding = blockSize - height % blockSize
        }
        Core.copyMakeBorder(
            source,
            source,
            0,
            bottomPadding,
            0,
            rightPadding,
            Core.BORDER_CONSTANT,
            Scalar.all(0.0)
        )
        return source
    }

    /**
     * calculate ridge segment by doing block process for the given image using the given block size.
     */
    private fun ridgeSegment(
        source: Mat,
        result: Mat,
        mask: Mat,
        blockSize: Int,
        threshold: Double
    ) {

        // for each block, get standard deviation
        // and replace the block with it
        val widthSteps = source.width() / blockSize
        val heightSteps = source.height() / blockSize
        val mean = MatOfDouble(0.toDouble())
        val std = MatOfDouble(0.toDouble())
        var window: Mat?
        val scalarBlack = Scalar.all(0.0)
        val scalarWhile = Scalar.all(255.0)
        val windowMask = Mat(source.rows(), source.cols(), CvType.CV_8UC1)
        var roi: Rect
        var stdVal: Double
        for (y in 1..heightSteps) {
            for (x in 1..widthSteps) {
                roi = Rect(blockSize * (x - 1), blockSize * (y - 1), blockSize, blockSize)
                windowMask.setTo(scalarBlack)
                Imgproc.rectangle(
                    windowMask, Point(roi.x.toDouble(), roi.y.toDouble()), Point(
                        (roi.x + roi.width).toDouble(), (roi.y + roi.height).toDouble()
                    ), scalarWhile, -1, 8, 0
                )
                window = source.submat(roi)
                Core.meanStdDev(window, mean, std)
                stdVal = std.toArray()[0]
                result.setTo(Scalar.all(stdVal), windowMask)

                // mask used to calc mean and standard deviation later
                mask.setTo(Scalar.all(if (stdVal >= threshold) 1.toDouble() else 0.toDouble()), windowMask)
            }
        }

        // get mean and standard deviation
        Core.meanStdDev(source, mean, std, mask)
        Core.subtract(source, Scalar.all(mean.toArray()[0]), result)
        Core.meanStdDev(result, mean, std, mask)
        Core.divide(result, Scalar.all(std.toArray()[0]), result)
    }

    /**
     * Calculate ridge orientation.
     */
    private fun ridgeOrientation(
        ridgeSegment: Mat,
        result: Mat,
        gradientSigma: Double,
        blockSigma: Double,
        orientSmoothSigma: Double
    ) {
        val rows = ridgeSegment.rows()
        val cols = ridgeSegment.cols()

        // calculate image gradients
        var kSize = Math.round((6 * gradientSigma).toFloat())
        if (kSize % 2 == 0) {
            kSize++
        }
        var kernel = gaussianKernel(kSize, gradientSigma)
        val fXKernel = Mat(1, 3, CvType.CV_32FC1)
        val fYKernel = Mat(3, 1, CvType.CV_32FC1)
        fXKernel.put(0, 0, -1.0)
        fXKernel.put(0, 1, 0.0)
        fXKernel.put(0, 2, 1.0)
        fYKernel.put(0, 0, -1.0)
        fYKernel.put(1, 0, 0.0)
        fYKernel.put(2, 0, 1.0)
        val fX = Mat(kSize, kSize, CvType.CV_32FC1)
        val fY = Mat(kSize, kSize, CvType.CV_32FC1)
        Imgproc.filter2D(kernel, fX, CvType.CV_32FC1, fXKernel)
        Imgproc.filter2D(kernel, fY, CvType.CV_32FC1, fYKernel)
        val gX = Mat(rows, cols, CvType.CV_32FC1)
        val gY = Mat(rows, cols, CvType.CV_32FC1)
        Imgproc.filter2D(ridgeSegment, gX, CvType.CV_32FC1, fX)
        Imgproc.filter2D(ridgeSegment, gY, CvType.CV_32FC1, fY)

        // covariance data for the image gradients
        val gXX = Mat(rows, cols, CvType.CV_32FC1)
        val gXY = Mat(rows, cols, CvType.CV_32FC1)
        val gYY = Mat(rows, cols, CvType.CV_32FC1)
        Core.multiply(gX, gX, gXX)
        Core.multiply(gX, gY, gXY)
        Core.multiply(gY, gY, gYY)

        // smooth the covariance data to perform a weighted summation of the data.
        kSize = Math.round((6 * blockSigma).toFloat())
        if (kSize % 2 == 0) {
            kSize++
        }
        kernel = gaussianKernel(kSize, blockSigma)
        Imgproc.filter2D(gXX, gXX, CvType.CV_32FC1, kernel)
        Imgproc.filter2D(gYY, gYY, CvType.CV_32FC1, kernel)
        Imgproc.filter2D(gXY, gXY, CvType.CV_32FC1, kernel)
        Core.multiply(gXY, Scalar.all(2.0), gXY)

        // analytic solution of principal direction
        val denom = Mat(rows, cols, CvType.CV_32FC1)
        val gXXMiusgYY = Mat(rows, cols, CvType.CV_32FC1)
        val gXXMiusgYYSquared = Mat(rows, cols, CvType.CV_32FC1)
        val gXYSquared = Mat(rows, cols, CvType.CV_32FC1)
        Core.subtract(gXX, gYY, gXXMiusgYY)
        Core.multiply(gXXMiusgYY, gXXMiusgYY, gXXMiusgYYSquared)
        Core.multiply(gXY, gXY, gXYSquared)
        Core.add(gXXMiusgYYSquared, gXYSquared, denom)
        Core.sqrt(denom, denom)

        // sine and cosine of doubled angles
        val sin2Theta = Mat(rows, cols, CvType.CV_32FC1)
        val cos2Theta = Mat(rows, cols, CvType.CV_32FC1)
        Core.divide(gXY, denom, sin2Theta)
        Core.divide(gXXMiusgYY, denom, cos2Theta)

        // smooth orientations (sine and cosine)
        // smoothed sine and cosine of doubled angles
        kSize = Math.round((6 * orientSmoothSigma).toFloat())
        if (kSize % 2 == 0) {
            kSize++
        }
        kernel = gaussianKernel(kSize, orientSmoothSigma)
        Imgproc.filter2D(sin2Theta, sin2Theta, CvType.CV_32FC1, kernel)
        Imgproc.filter2D(cos2Theta, cos2Theta, CvType.CV_32FC1, kernel)

        // calculate the result as the following, so the values of the matrix range [0, PI]
        //orientim = atan2(sin2theta,cos2theta)/360;
        atan2(sin2Theta, cos2Theta, result)
        Core.multiply(result, Scalar.all(Math.PI / 360.0), result)
    }

    /**
     * Create Gaussian kernel.
     */
    private fun gaussianKernel(kSize: Int, sigma: Double): Mat {
        val kernelX = Imgproc.getGaussianKernel(kSize, sigma, CvType.CV_32FC1)
        val kernelY = Imgproc.getGaussianKernel(kSize, sigma, CvType.CV_32FC1)
        val kernel = Mat(kSize, kSize, CvType.CV_32FC1)
        Core.gemm(
            kernelX,
            kernelY.t(),
            1.0,
            Mat.zeros(kSize, kSize, CvType.CV_32FC1),
            0.0,
            kernel,
            0
        )
        return kernel
    }

    /**
     * Calculate bitwise atan2 for the given 2 images.
     */
    private fun atan2(src1: Mat, src2: Mat, dst: Mat) {
        val height = src1.height()
        val width = src2.width()
        for (y in 0 until height) {
            for (x in 0 until width) {
                dst.put(
                    y, x, Core.fastAtan2(
                        src1[y, x][0].toFloat(), src2[y, x][0].toFloat()
                    ).toDouble()
                )
            }
        }
    }

    /**
     * Calculate ridge frequency.
     */
    private fun ridgeFrequency(
        ridgeSegment: Mat,
        segmentMask: Mat,
        ridgeOrientation: Mat,
        frequencies: Mat,
        blockSize: Int,
        windowSize: Int,
        minWaveLength: Int,
        maxWaveLength: Int
    ): Double {
        val rows = ridgeSegment.rows()
        val cols = ridgeSegment.cols()
        var blockSegment: Mat
        var blockOrientation: Mat
        var frequency: Mat
        var y = 0
        while (y < rows - blockSize) {
            var x = 0
            while (x < cols - blockSize) {
                blockSegment = ridgeSegment.submat(y, y + blockSize, x, x + blockSize)
                blockOrientation = ridgeOrientation.submat(y, y + blockSize, x, x + blockSize)
                frequency = calculateFrequency(
                    blockSegment,
                    blockOrientation,
                    windowSize,
                    minWaveLength,
                    maxWaveLength
                )
                frequency.copyTo(frequencies.rowRange(y, y + blockSize).colRange(x, x + blockSize))
                x += blockSize
            }
            y += blockSize
        }

        // mask out frequencies calculated for non ridge regions
        Core.multiply(frequencies, segmentMask, frequencies, 1.0, CvType.CV_32FC1)

        // find median frequency over all the valid regions of the image.
        val medianFrequency = medianFrequency(frequencies)

        // the median frequency value used across the whole fingerprint gives a more satisfactory result
        Core.multiply(segmentMask, Scalar.all(medianFrequency), frequencies, 1.0, CvType.CV_32FC1)
        return medianFrequency
    }

    /**
     * Estimate fingerprint ridge frequency within image block.
     */
    private fun calculateFrequency(
        block: Mat,
        blockOrientation: Mat,
        windowSize: Int,
        minWaveLength: Int,
        maxWaveLength: Int
    ): Mat {
        val rows = block.rows()
        val cols = block.cols()
        val orientation = blockOrientation.clone()
        Core.multiply(orientation, Scalar.all(2.0), orientation)
        val orientLength = orientation.total().toInt()
        val orientations = FloatArray(orientLength)
        orientation[0, 0, orientations]
        val sinOrient = DoubleArray(orientLength)
        val cosOrient = DoubleArray(orientLength)
        for (i in 1 until orientLength) {
            sinOrient[i] = Math.sin(orientations[i].toDouble())
            cosOrient[i] = Math.cos(orientations[i].toDouble())
        }
        val orient = Core.fastAtan2(
            calculateMean(sinOrient).toFloat(),
            calculateMean(cosOrient).toFloat()
        ) / 2.0.toFloat()

        // rotate the image block so that the ridges are vertical
        val rotated = Mat(rows, cols, CvType.CV_32FC1)
        val center = Point((cols / 2).toDouble(), (rows / 2).toDouble())
        val rotateAngle = orient / Math.PI * 180.0 + 90.0
        val rotateScale = 1.0
        val rotatedSize = Size(cols.toDouble(), rows.toDouble())
        val rotateMatrix = Imgproc.getRotationMatrix2D(center, rotateAngle, rotateScale)
        Imgproc.warpAffine(block, rotated, rotateMatrix, rotatedSize, Imgproc.INTER_NEAREST)

        // crop the image so that the rotated image does not contain any invalid regions
        // this prevents the projection down the columns from being mucked up
        val cropSize = Math.round(rows / Math.sqrt(2.0)).toInt()
        val offset = Math.round((rows - cropSize) / 2.0).toInt() - 1
        val cropped = rotated.submat(offset, offset + cropSize, offset, offset + cropSize)

        // get sums of columns
        var sum: Float
        val proj = Mat(1, cropped.cols(), CvType.CV_32FC1)
        for (c in 1 until cropped.cols()) {
            sum = 0f
            for (r in 1 until cropped.cols()) {
                sum = (sum + cropped[r, c][0]).toFloat()
            }
            proj.put(0, c, sum.toDouble())
        }

        // find peaks in projected grey values by performing a grayScale
        // dilation and then finding where the dilation equals the original values.
        val dilateKernel = Mat(windowSize, windowSize, CvType.CV_32FC1, Scalar.all(1.0))
        val dilate = Mat(1, cropped.cols(), CvType.CV_32FC1)
        Imgproc.dilate(proj, dilate, dilateKernel, Point(-1.toDouble(), -1.toDouble()), 1)
        //Imgproc.dilate(proj, dilate, dilateKernel, new Point(-1, -1), 1, Imgproc.BORDER_CONSTANT, Scalar.all(0.0));
        val projMean = Core.mean(proj).`val`[0]
        var projValue: Double
        var dilateValue: Double
        val ROUND_POINTS = 1000.0
        val maxind = ArrayList<Int>()
        for (i in 0 until cropped.cols()) {
            projValue = proj[0, i][0]
            dilateValue = dilate[0, i][0]

            // round to maximize the likelihood of equality
            projValue = Math.round(projValue * ROUND_POINTS).toDouble() / ROUND_POINTS
            dilateValue = Math.round(dilateValue * ROUND_POINTS).toDouble() / ROUND_POINTS
            if (dilateValue == projValue && projValue > projMean) {
                maxind.add(i)
            }
        }

        // determine the spatial frequency of the ridges by dividing the distance between
        // the 1st and last peaks by the (No of peaks-1). If no peaks are detected
        // or the wavelength is outside the allowed bounds, the frequency image is set to 0
        var result = Mat(rows, cols, CvType.CV_32FC1, Scalar.all(0.0))
        val peaks = maxind.size
        if (peaks >= 2) {
            val waveLength = ((maxind[peaks - 1] - maxind[0]) / (peaks - 1)).toDouble()
            if (waveLength >= minWaveLength && waveLength <= maxWaveLength) {
                result = Mat(rows, cols, CvType.CV_32FC1, Scalar.all(1.0 / waveLength))
            }
        }
        return result
    }

    /**
     * Enhance fingerprint image using oriented filters.
     */
    private fun ridgeFilter(
        ridgeSegment: Mat,
        orientation: Mat,
        frequency: Mat,
        result: Mat,
        kx: Double,
        ky: Double,
        medianFreq: Double
    ): Int {
        val angleInc = 3
        val rows = ridgeSegment.rows()
        val cols = ridgeSegment.cols()
        val filterCount = 180 / angleInc
        val filters = arrayOfNulls<Mat>(filterCount)
        val sigmaX = kx / medianFreq
        val sigmaY = ky / medianFreq

        //mat refFilter = exp(-(x. ^ 2 / sigmaX ^ 2 + y. ^ 2 / sigmaY ^ 2) / 2). * cos(2 * pi * medianFreq * x);
        var size = Math.round(3 * Math.max(sigmaX, sigmaY)).toInt()
        size = if (size % 2 == 0) size else size + 1
        val length = size * 2 + 1
        val x = meshGrid(size)
        val y = x.t()
        val xSquared = Mat(length, length, CvType.CV_32FC1)
        val ySquared = Mat(length, length, CvType.CV_32FC1)
        Core.multiply(x, x, xSquared)
        Core.multiply(y, y, ySquared)
        Core.divide(xSquared, Scalar.all(sigmaX * sigmaX), xSquared)
        Core.divide(ySquared, Scalar.all(sigmaY * sigmaY), ySquared)
        val refFilterPart1 = Mat(length, length, CvType.CV_32FC1)
        Core.add(xSquared, ySquared, refFilterPart1)
        Core.divide(refFilterPart1, Scalar.all(-2.0), refFilterPart1)
        Core.exp(refFilterPart1, refFilterPart1)
        var refFilterPart2 = Mat(length, length, CvType.CV_32FC1)
        Core.multiply(x, Scalar.all(2 * Math.PI * medianFreq), refFilterPart2)
        refFilterPart2 = matCos(refFilterPart2)
        val refFilter = Mat(length, length, CvType.CV_32FC1)
        Core.multiply(refFilterPart1, refFilterPart2, refFilter)

        // Generate rotated versions of the filter.  Note orientation
        // image provides orientation *along* the ridges, hence +90
        // degrees, and the function requires angles +ve anticlockwise, hence the minus sign.
        var rotated: Mat
        var rotateMatrix: Mat?
        var rotateAngle: Double
        val center = Point((length / 2).toDouble(), (length / 2).toDouble())
        val rotatedSize = Size(length.toDouble(), length.toDouble())
        val rotateScale = 1.0
        for (i in 0 until filterCount) {
            rotateAngle = -(i * angleInc).toDouble()
            rotated = Mat(length, length, CvType.CV_32FC1)
            rotateMatrix = Imgproc.getRotationMatrix2D(center, rotateAngle, rotateScale)
            Imgproc.warpAffine(refFilter, rotated, rotateMatrix, rotatedSize, Imgproc.INTER_LINEAR)
            filters[i] = rotated
        }

        // convert orientation matrix values from radians to an index value
        // that corresponds to round(degrees/angleInc)
        val orientIndexes = Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1)
        Core.multiply(
            orientation,
            Scalar.all(filterCount.toDouble() / Math.PI),
            orientIndexes,
            1.0,
            CvType.CV_8UC1
        )
        var orientMask: Mat
        var orientThreshold: Mat
        orientMask = Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1, Scalar.all(0.0))
        orientThreshold =
            Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1, Scalar.all(0.0))
        Core.compare(orientIndexes, orientThreshold, orientMask, Core.CMP_LT)
        Core.add(orientIndexes, Scalar.all(filterCount.toDouble()), orientIndexes, orientMask)
        orientMask = Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1, Scalar.all(0.0))
        orientThreshold = Mat(
            orientation.rows(),
            orientation.cols(),
            CvType.CV_8UC1,
            Scalar.all(filterCount.toDouble())
        )
        Core.compare(orientIndexes, orientThreshold, orientMask, Core.CMP_GE)
        Core.subtract(orientIndexes, Scalar.all(filterCount.toDouble()), orientIndexes, orientMask)

        // finally, find where there is valid frequency data then do the filtering
        val value = Mat(length, length, CvType.CV_32FC1)
        var subSegment: Mat?
        var orientIndex: Int
        var sum: Double
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                if (frequency[r, c][0] > 0 && r > size + 1 && r < rows - size - 1 && c > size + 1 && c < cols - size - 1) {
                    orientIndex = orientIndexes[r, c][0].toInt()
                    subSegment = ridgeSegment.submat(r - size - 1, r + size, c - size - 1, c + size)
                    Core.multiply(subSegment, filters[orientIndex], value)
                    sum = Core.sumElems(value).`val`[0]
                    result.put(r, c, sum)
                }
            }
        }
        return size
    }

    /**
     * Enhance the image after ridge filter.
     * Apply mask, binary threshold, thinning, ..., etc.
     */
    private fun enhancement(
        source: Mat,
        result: Mat,
        blockSize: Int,
        rows: Int,
        cols: Int,
        padding: Int
    ) {
        val MatSnapShotMask = snapShotMask(rows, cols, padding)
        val paddedMask = imagePadding(MatSnapShotMask, blockSize)
        if (BuildConfig.DEBUG && paddedMask.size() != source.size()) {
            throw RuntimeException("Incompatible sizes of image and mask")
        }

        // apply the original mask to get rid of extras
        Core.multiply(source, paddedMask, result, 1.0, CvType.CV_8UC1)

        // apply binary threshold
        Imgproc.threshold(result, result, 0.0, 255.0, Imgproc.THRESH_BINARY)
    }

    /**
     * Create mesh grid.
     */
    private fun meshGrid(size: Int): Mat {
        val l = size * 2 + 1
        var value = -size
        val result = Mat(l, l, CvType.CV_32FC1)
        for (c in 0 until l) {
            for (r in 0 until l) {
                result.put(r, c, value.toDouble())
            }
            value++
        }
        return result
    }

    /**
     * Apply cos to each element of the matrix.
     */
    private fun matCos(source: Mat): Mat {
        val rows = source.rows()
        val cols = source.cols()
        val result = Mat(cols, rows, CvType.CV_32FC1)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                result.put(r, c, Math.cos(source[r, c][0]))
            }
        }
        return result
    }

    /**
     * Calculate the median of all values greater than zero.
     */
    private fun medianFrequency(image: Mat): Double {
        val values = ArrayList<Double>()
        var value: Double
        for (r in 0 until image.rows()) {
            for (c in 0 until image.cols()) {
                value = image[r, c][0]
                if (value > 0) {
                    values.add(value)
                }
            }
        }
        Collections.sort(values)
        val size = values.size
        var median = 0.0
        if (size > 0) {
            val halfSize = size / 2
            median = if (size % 2 == 0) {
                (values[halfSize - 1] + values[halfSize]) / 2.0
            } else {
                values[halfSize]
            }
        }
        return median
    }

    /**
     * Calculate mean of given array.
     */
    private fun calculateMean(m: DoubleArray): Double {
        var sum = 0.0
        for (aM in m) {
            sum += aM
        }
        return sum / m.size
    }

    /**
     * Mask used in the snapshot.
     */
    private fun snapShotMask(rows: Int, cols: Int, padding: Int): Mat {
        /*
        Some magic numbers. We have no idea where these come from?!
        int maskWidth = 260;
        int maskHeight = 160;
        */
        val center = Point((cols / 2).toDouble(), (rows / 2).toDouble())
        val axes = Size((cols / 2 - padding).toDouble(), (rows / 2 - padding).toDouble())
        val scalarWhite = Scalar(255.toDouble(), 255.toDouble(), 255.toDouble())
        val scalarBlack = Scalar(0.toDouble(), 0.toDouble(), 0.toDouble())
        val thickness = -1
        val lineType = 8
        val mask = Mat(rows, cols, CvType.CV_8UC1, scalarBlack)
        Imgproc.ellipse(mask, center, axes, 0.0, 0.0, 360.0, scalarWhite, thickness, lineType, 0)
        return mask
    }

    private fun thinning(img: Mat): Mat {
        var thinned = Mat(img.size(), CvType.CV_8UC1)
        Imgproc.threshold(img, thinned, 0.0, 255.0, Imgproc.THRESH_OTSU)
        val t = Thinning()
        thinned = t.doJaniThinning(thinned)
        return thinned
    }

    companion object {
        val TAG = ImageProcessing::class.java.simpleName

        const val ACTION_STEP_START_FINGERPRINT_EXTRACTION = "example.jllarraz.com.myapplication.STEP_START_FINGERPRINT_EXTRACTION"
        const val ACTION_STEP_SKIN_DETECTION = "example.jllarraz.com.myapplication.STEP_SKIN_DETECTION"
        const val ACTION_STEP_HISTOGRAM_EQUALIZATION = "example.jllarraz.com.myapplication.STEP_HISTOGRAM_EQUALIZATION"
        const val ACTION_STEP_CLAHE_EQUALIZATION = "example.jllarraz.com.myapplication.ACTION_STEP_CLAHE_EQUALIZATION"
        const val ACTION_STEP_ADAPTATIVE_THRESHOLD = "example.jllarraz.com.myapplication.ACTION_STEP_ADAPTATIVE_THRESHOLD"
        const val ACTION_STEP_BILATERAL_FILTER = "example.jllarraz.com.myapplication.ACTION_STEP_BILATERAL_FILTER"
        const val ACTION_STEP_FINGERPRINT_SKELETIZATION = "example.jllarraz.com.myapplication.STEP_FINGERPRINT_SKELETIZATION"
        const val ACTION_STEP_RIDGE_THINNING = "example.jllarraz.com.myapplication.STEP_RIDGE_THINNING"
        const val ACTION_STEP_MINUTIAE_EXTRATION = "example.jllarraz.com.myapplication.STEP_MINUTIAE_EXTRATION"
        const val ACTION_STEP_FINISH_FINGERPRINT_EXTRACTION = "example.jllarraz.com.myapplication.STEP_FINISH_FINGERPRINT_EXTRACTION"



        var keypoints: MatOfKeyPoint? = null
            private set
        var descriptors: Mat? = null
            private set

        /**
         * Match using the ratio test and RANSAC.
         * Returns the ratio matches and the total keypoints
         */
        fun matchFeatures(keyPoints: MatOfKeyPoint, descriptors: Mat?): Double {
            val matcher = BFMatcher.create(Core.NORM_HAMMING, true)
            val matches = MatOfDMatch()
            matcher.match(Companion.descriptors, descriptors, matches)
            val maxDistance = 100
            var matchCount = 0
            for (m in matches.toArray()) {
                if (m.distance <= maxDistance) matchCount++
            }
            return (matchCount.toFloat() / Math.max(
                keyPoints.rows(),
                keypoints!!.rows()
            )).toDouble()
        }

        fun preprocess(frame: Mat, width: Int, height: Int): Bitmap {
            // convert to grayscale
            val frameGrey = Mat(height, width, CvType.CV_8UC1)
            Imgproc.cvtColor(frame, frameGrey, Imgproc.COLOR_BGR2GRAY, 1)

            // rotate
            val rotatedFrame = Mat(width, height, frameGrey.type())
            Core.transpose(frameGrey, rotatedFrame)
            Core.flip(rotatedFrame, rotatedFrame, Core.ROTATE_180)

            // resize to match the surface view
            val resizedFrame = Mat(width, height, rotatedFrame.type())
            Imgproc.resize(rotatedFrame, resizedFrame, Size(width.toDouble(), height.toDouble()))

            // crop
            val ellipseMask = getEllipseMask(width, height)
            val frameCropped =
                Mat(resizedFrame.rows(), resizedFrame.cols(), resizedFrame.type(), Scalar(0.toDouble()))
            resizedFrame.copyTo(frameCropped, ellipseMask)

            // histogram equalisation
            val frameHistEq = Mat(frame.rows(), frameCropped.cols(), frameCropped.type())
            Imgproc.equalizeHist(frameCropped, frameHistEq)

            // convert back to rgba
            val frameRgba = Mat(frameHistEq.rows(), frameHistEq.cols(), CvType.CV_8UC4)
            Imgproc.cvtColor(frameHistEq, frameRgba, Imgproc.COLOR_GRAY2RGBA)

            // crop again to correct alpha
            val frameAlpha =
                Mat(frameRgba.rows(), frameRgba.cols(), CvType.CV_8UC4, Scalar(0.toDouble(), 0.toDouble(), 0.toDouble(), 0.toDouble()))
            frameRgba.copyTo(frameAlpha, ellipseMask)

            // convert to bitmap
            val bmp =
                Bitmap.createBitmap(frameAlpha.cols(), frameAlpha.rows(), Bitmap.Config.ARGB_4444)
            Utils.matToBitmap(frameAlpha, bmp)
            return bmp
        }

        private var ellipseMask: Mat? = null
        private fun getEllipseMask(width: Int, height: Int): Mat {
            if (ellipseMask == null || ellipseMask!!.cols() != width || ellipseMask!!.rows() != height) {
                val paddingX = (CameraOverlayView.PADDING * width.toFloat()).toInt()
                val paddingY = (CameraOverlayView.PADDING * height.toFloat()).toInt()
                val box = RotatedRect(
                    Point((width / 2).toDouble(), (height / 2).toDouble()),
                    Size((width - 2 * paddingX).toDouble(), (height - 2 * paddingY).toDouble()),
                    0.toDouble()
                )
                ellipseMask = Mat(height, width, CvType.CV_8UC1, Scalar(0.toDouble()))
                Imgproc.ellipse(ellipseMask, box, Scalar(255.toDouble()), -1)
            }
            return ellipseMask!!
        }
    }
}

internal class Thinning {
    private lateinit var B: Array<BooleanArray>
    fun doJaniThinning(Image: Mat): Mat {
        B = Array(Image.rows()) { BooleanArray(Image.cols()) }
        // Inverse of B
        val B_ = Array(Image.rows()) { BooleanArray(Image.cols()) }
        for (i in 0 until Image.rows()) for (j in 0 until Image.cols()) B[i][j] =
            Image[i, j][0] > 10 //not a mistake, in matlab first invert and then morph
        val prevB = Array(Image.rows()) { BooleanArray(Image.cols()) }
        val maxIter = 1000
        for (iter in 0 until maxIter) {
            // Assign B to prevB
            for (i in 0 until Image.rows()) System.arraycopy(B[i], 0, prevB[i], 0, Image.cols())

            //Iteration #1
            for (i in 0 until Image.rows()) for (j in 0 until Image.cols()) B_[i][j] =
                !(B[i][j] && G1(i, j) && G2(i, j) && G3(i, j)) && B[i][j]

            // Assign result of iteration #1 to B, so that iteration #2 will see the results
            for (i in 0 until Image.rows()) System.arraycopy(B_[i], 0, B[i], 0, Image.cols())


            //Iteration #2
            for (i in 0 until Image.rows()) for (j in 0 until Image.cols()) B_[i][j] =
                !(B[i][j] && G1(i, j) && G2(i, j) && G3_(i, j)) && B[i][j]

            // Assign result of Iteration #2 to B
            for (i in 0 until Image.rows()) System.arraycopy(B_[i], 0, B[i], 0, Image.cols())

            // stop when it doesn't change anymore
            var convergence = true
            for (i in 0 until Image.rows()) convergence =
                convergence and Arrays.equals(B[i], prevB[i])
            if (convergence) {
                break
            }
        }
        removeFalseRidgeEndings(Image)
        val r = Mat.zeros(Image.size(), CvType.CV_8UC1)
        for (i in 0 until Image.rows()) for (j in 0 until Image.cols()) if (B[i][j]) r.put(
            i,
            j,
            255.0
        )
        return r
    }

    // remove ridge endings shorter than minimumRidgeLength
    private fun removeFalseRidgeEndings(Image: Mat) {
        val minimumRidgeLength = 5
        for (i in 0 until Image.rows()) for (j in 0 until Image.cols()) if (B[i][j] && neighbourCount(
                i,
                j
            ) == 1
        ) removeEnding(i, j, minimumRidgeLength)
    }

    // follow ridge recursively and remove if shorter than minimumlength
    private fun removeEnding(i: Int, j: Int, minimumLength: Int): Boolean {
        if (minimumLength < 0) return true
        if (neighbourCount(i, j) > 1) return false
        B[i][j] = false
        if (neighbourCount(i, j) == 0) return false
        var index = 0
        for (a in 1..8) {
            if (x(a, i, j)) {
                index = a
                break
            }
        }
        var _i = i
        var _j = j
        when (index) {
            1 -> _i = i + 1
            2 -> {
                _i = i + 1
                _j = j + 1
            }
            3 -> _j = j + 1
            4 -> {
                _i = i - 1
                _j = j + 1
            }
            5 -> _i = i - 1
            6 -> {
                _i = i - 1
                _j = j - 1
            }
            7 -> _j = j - 1
            8 -> {
                _i = i + 1
                _j = j - 1
            }
        }
        val ok = removeEnding(_i, _j, minimumLength - 1)
        if (ok) B[i][j] = true
        return ok
    }

    private fun neighbourCount(i: Int, j: Int): Int {
        var cn = 0
        for (a in 1..8) if (x(a, i, j)) cn++
        return cn
    }

    private fun x(a: Int, i: Int, j: Int): Boolean {
        try {
            when (a) {
                1 -> return B[i + 1][j]
                2 -> return B[i + 1][j + 1]
                3 -> return B[i][j + 1]
                4 -> return B[i - 1][j + 1]
                5 -> return B[i - 1][j]
                6 -> return B[i - 1][j - 1]
                7 -> return B[i][j - 1]
                8 -> return B[i + 1][j - 1]
            }
        } catch (e: IndexOutOfBoundsException) {
            return false
        }
        return false
    }

    private fun G1(i: Int, j: Int): Boolean {
        var X = 0
        for (q in 1..4) {
            if (!x(2 * q - 1, i, j) && (x(2 * q, i, j) || x(2 * q + 1, i, j))) X++
        }
        return X == 1
    }

    private fun G2(i: Int, j: Int): Boolean {
        val m = Math.min(n1(i, j), n2(i, j))
        return m == 2 || m == 3
    }

    private fun n1(i: Int, j: Int): Int {
        var r = 0
        for (q in 1..4) if (x(2 * q - 1, i, j) || x(2 * q, i, j)) r++
        return r
    }

    private fun n2(i: Int, j: Int): Int {
        var r = 0
        for (q in 1..4) if (x(2 * q, i, j) || x(2 * q + 1, i, j)) r++
        return r
    }

    private fun G3(i: Int, j: Int): Boolean {
        return (x(2, i, j) || x(3, i, j) || !x(8, i, j)) && x(1, i, j)
    }

    private fun G3_(i: Int, j: Int): Boolean {
        return (x(6, i, j) || x(7, i, j) || !x(4, i, j)) && x(5, i, j)
    }
}

internal class Minutiae(var x: Int, var y: Int, var type: Type) {
    internal enum class Type {
        BIFURCATION, RIDGEENDING
    }

    fun euclideanDistance(m: Minutiae): Double {
        return Math.sqrt(Math.pow((x - m.x).toDouble(), 2.0) + Math.pow((y - m.y).toDouble(), 2.0))
    }

    companion object {
        const val BIFURCATION_LABEL = 1
        const val RIDGE_ENDING_LABEL = 0
    }
}