/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package example.jllarraz.com.minutiae.ui.fragments

import android.Manifest
import android.annotation.SuppressLint
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.hardware.display.DisplayManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.provider.Settings
import android.util.DisplayMetrics
import android.util.Log
import android.view.ScaleGestureDetector
import android.view.Surface
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.internal.Camera2CameraFactory
import androidx.camera.camera2.internal.Camera2CameraInfoImpl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.*
import androidx.camera.core.Camera
//import androidx.camera.extensions.AutoPreviewExtender
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.LiveData
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.google.mlkit.vision.common.InputImage
import example.jllarraz.com.minutiae.permission.PermissionStatus
import example.jllarraz.com.minutiae.utils.PermissionUtils
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.collections.ArrayList
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import example.jllarraz.com.myapplication.R
import androidx.camera.core.FocusMeteringAction







/** Helper type alias used for analysis use case callbacks */
typealias LumaListener = (luma: Double) -> Unit

/**
 * Main fragment for this app. Implements all camera operations including:
 * - Viewfinder
 * - Photo taking
 * - Image analysis
 */
abstract class CameraXFragment : Fragment() {


    var isAutoCameraPermissionRevokedHandled:Boolean = true

    val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
            onCallbackRequestPermissions(results)
        }

    val registerActivitySettingsForResult =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            onCallbackRequestActivitySettings(result)
        }
    
    private lateinit var broadcastManager: LocalBroadcastManager

    private var displayId: Int = -1
    protected var cameraId: Int = -1
    protected var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    protected var isAutoFocusEnable = false
    protected var isTorchEnable = false


    //abstract val callbackFrameProcessor: FrameProcessor
    abstract val cameraPreview: PreviewView
    abstract val requestedPermissions:ArrayList<String>

    private val displayManager by lazy {
        requireContext().getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
    }

    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService

    /**
     * We need a display listener for orientation changes that do not trigger a configuration
     * change, for example if we choose to override config change in manifest or for 180-degree
     * orientation changes.
     */
    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) = Unit
        override fun onDisplayRemoved(displayId: Int) = Unit
        override fun onDisplayChanged(displayId: Int) = view?.let { view ->
            if (displayId == this@CameraXFragment.displayId) {
                Log.d(TAG, "Rotation changed: ${view.display.rotation}")
                imageCapture?.targetRotation = view.display.rotation
                imageAnalyzer?.targetRotation = view.display.rotation
            }
        } ?: Unit
    }

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.

        checkPermissions(requestedPermissions)

        Handler().postDelayed(Runnable {
            enableAutoFocus(isAutoFocusEnable)
            enableTorch(isTorchEnable)
        }, 300)

    }

    override fun onDestroyView() {
        super.onDestroyView()

        // Shut down our background executor
        cameraExecutor.shutdown()

        // Unregister the broadcast receivers and listeners
        displayManager.unregisterDisplayListener(displayListener)
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        if(savedInstanceState!=null){
            if(savedInstanceState.containsKey(KEY_AUTO_FOCUS)){
                isAutoFocusEnable = savedInstanceState.getBoolean(KEY_AUTO_FOCUS)
            }
            if(savedInstanceState.containsKey(KEY_IS_TORCH_ENABLE)){
                isTorchEnable = savedInstanceState.getBoolean(KEY_IS_TORCH_ENABLE)
            }

            if (savedInstanceState.containsKey(KEY_AUTO_HANDLE_CAMERA_PERMISSION_REVOKED)) {
                isAutoCameraPermissionRevokedHandled = savedInstanceState.getBoolean(
                    KEY_AUTO_HANDLE_CAMERA_PERMISSION_REVOKED, true)
            }
        }

        // Initialize our background executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        broadcastManager = LocalBroadcastManager.getInstance(view.context)

        // Every time the orientation of device changes, update rotation for use cases
        displayManager.registerDisplayListener(displayListener, null)

        // Determine the output directory


        /*// Wait for the views to be properly laid out
        cameraPreview.post {

            // Keep track of the display in which this view is attached
            displayId = cameraPreview.display.displayId

            // Set up the camera and its use cases
            setUpCamera()
        }*/

        val scaleGestureDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.OnScaleGestureListener {
            override fun onScaleBegin(detector: ScaleGestureDetector?): Boolean {
                return true
            }

            override fun onScaleEnd(detector: ScaleGestureDetector?) {
            }

            override fun onScale(detector: ScaleGestureDetector?): Boolean {
                val maxZoomRatio = camera?.cameraInfo?.zoomState?.value?.maxZoomRatio
                val minZoomRatio = camera?.cameraInfo?.zoomState?.value?.minZoomRatio

                val zoomRatio = camera?.cameraInfo?.zoomState?.value?.zoomRatio
                Log.d(TAG, "Max zoom ratio: " + maxZoomRatio)
                Log.d(TAG, "Min zoom ratio: " + minZoomRatio)
                Log.d(TAG, "Zoom ratio: " + zoomRatio)

                zoomRatio?.let {
                    val scale = it * detector?.scaleFactor!!
                    camera?.cameraControl?.setZoomRatio(scale)
                    //camera!!.cameraControl.setLinearZoom(1F)
                }
                return true
            }

        })

        cameraPreview.setOnTouchListener { _, event ->
            scaleGestureDetector.onTouchEvent(event)
            return@setOnTouchListener true
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean(KEY_AUTO_FOCUS, isAutoFocusEnable)
        outState.putBoolean(KEY_IS_TORCH_ENABLE, isTorchEnable)
        outState.putBoolean(KEY_AUTO_HANDLE_CAMERA_PERMISSION_REVOKED, isAutoCameraPermissionRevokedHandled)
        super.onSaveInstanceState(outState)
    }

    /**
     * Inflate camera controls and update the UI manually upon config changes to avoid removing
     * and re-adding the view finder from the view hierarchy; this provides a seamless rotation
     * transition on devices that support it.
     *
     * NOTE: The flag is supported starting in Android 8 but there still is a small flash on the
     * screen for devices that run Android 9 or below.
     */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
    }

    /** Initialize CameraX, and prepare to bind the camera use cases  */
    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(Runnable {

            // CameraProvider
            cameraProvider = cameraProviderFuture.get()

            // Select lensFacing depending on the available cameras
            lensFacing = when {
                hasBackCamera() -> CameraSelector.LENS_FACING_BACK
                hasFrontCamera() -> CameraSelector.LENS_FACING_FRONT
                else -> throw IllegalStateException("Back and front camera are unavailable")
            }

            // Build and bind the camera use cases
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    /** Declare and bind preview, capture and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError", "RestrictedApi", "UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // Get screen metrics used to setup camera for full screen resolution
        val metrics = DisplayMetrics().also { cameraPreview.display.getRealMetrics(it) }
        Log.d(TAG, "Screen metrics: ${metrics.widthPixels} x ${metrics.heightPixels}")

        val screenAspectRatio = aspectRatio(metrics.widthPixels, metrics.heightPixels)
        Log.d(TAG, "Preview aspect ratio: $screenAspectRatio")

        val rotation = cameraPreview.display.rotation

        // CameraProvider
        val cameraProvider = cameraProvider
                ?: throw IllegalStateException("Camera initialization failed.")

        // CameraSelector
        val cameraFilter = CameraFilter { cameraInfos ->
            val filteredCameraInfos = java.util.ArrayList<CameraInfo>()

            for (cameraInfo in cameraInfos) {
                if(cameraInfo is Camera2CameraInfoImpl){
                    cameraInfo.cameraId
                    if (cameraInfo.cameraId.equals(cameraId.toString())) {
                        filteredCameraInfos.add(cameraInfo)
                    }
                }
            }
            filteredCameraInfos
        }

        val cameraSelector:CameraSelector
        if(cameraId>-1){
            cameraSelector = CameraSelector.Builder().addCameraFilter (cameraFilter).build()
        } else {
            cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        }
        //val cameraSelector = CameraSelector.Builder().addCameraFilter (createCameraFilter).build()


        // Preview
        val previewBuilder = Preview.Builder()
        previewBuilder
                // We request aspect ratio but no resolution
                .setTargetAspectRatio(screenAspectRatio)
                // Set initial target rotation
                .setTargetRotation(rotation)

        //setFocusDistance(previewBuilder, 0f)


       /* val autoPreviewExtender = AutoPreviewExtender.create(previewBuilder)
        if(autoPreviewExtender.isExtensionAvailable(cameraSelector)){
            autoPreviewExtender.enableExtension(cameraSelector)
        }*/
        preview = previewBuilder.build()

        imageCapture = ImageCapture.Builder()
            .setTargetAspectRatio(screenAspectRatio)
            // Set initial target rotation, we will have to call this again if rotation changes
            // during the lifecycle of this use case
            .setTargetRotation(rotation)
            .build()

        // ImageAnalysis


        imageAnalyzer = ImageAnalysis.Builder()
                // We request aspect ratio but no resolution
                .setTargetAspectRatio(screenAspectRatio)
                // Set initial target rotation, we will have to call this again if rotation changes
                // during the lifecycle of this use case
                .setTargetRotation(rotation)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { imageProxy ->
                        // Values returned from our analyzer are passed to the attached listener
                        // We log image analysis results here - you should do something useful
                        // instead!
                        onImageProxy(imageProxy)
                        // Log.d(TAG, "Average luminosity: $imageProxy")
                    })
                }


        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            val useCases = ArrayList<UseCase>()
            imageCapture?.let { useCases.add(it) }
            imageAnalyzer?.let { useCases.add(it) }

            if(useCases.isNotEmpty()) {
                val useCasesArray = useCases.toTypedArray()
                camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, *useCasesArray)
            } else {
                camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview)
            }
            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(cameraPreview.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    /**
     *  [androidx.camera.core.ImageAnalysisConfig] requires enum value of
     *  [androidx.camera.core.AspectRatio]. Currently it has values of 4:3 & 16:9.
     *
     *  Detecting the most suitable ratio for dimensions provided in @params by counting absolute
     *  of preview ratio to one of the provided values.
     *
     *  @param width - preview width
     *  @param height - preview height
     *  @return suitable aspect ratio
     */
    private fun aspectRatio(width: Int, height: Int): Int {
        val previewRatio = max(width, height).toDouble() / min(width, height)
        if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
            return AspectRatio.RATIO_4_3
        }
        return AspectRatio.RATIO_16_9
    }

    /** Returns true if the device has an available back camera. False otherwise */
    private fun hasBackCamera(): Boolean {
        return cameraProvider?.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) ?: false
    }

    /** Returns true if the device has an available front camera. False otherwise */
    private fun hasFrontCamera(): Boolean {
        return cameraProvider?.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) ?: false
    }

    /**
     * Our custom image analysis class.
     *
     * <p>All we need to do is override the function `analyze` with our desired operations. Here,
     * we compute the average luminosity of the image by looking at the Y plane of the YUV frame.
     */
    private class LuminosityAnalyzer(listener: LumaListener? = null) : ImageAnalysis.Analyzer {
        private val frameRateWindow = 8
        private val frameTimestamps = ArrayDeque<Long>(5)
        private val listeners = ArrayList<LumaListener>().apply { listener?.let { add(it) } }
        private var lastAnalyzedTimestamp = 0L
        var framesPerSecond: Double = -1.0
            private set

        /**
         * Used to add listeners that will be called with each luma computed
         */
        fun onFrameAnalyzed(listener: LumaListener) = listeners.add(listener)

        /**
         * Helper extension function used to extract a byte array from an image plane buffer
         */
        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()    // Rewind the buffer to zero
            val data = ByteArray(remaining())
            get(data)   // Copy the buffer into a byte array
            return data // Return the byte array
        }

        /**
         * Analyzes an image to produce a result.
         *
         * <p>The caller is responsible for ensuring this analysis method can be executed quickly
         * enough to prevent stalls in the image acquisition pipeline. Otherwise, newly available
         * images will not be acquired and analyzed.
         *
         * <p>The image passed to this method becomes invalid after this method returns. The caller
         * should not store external references to this image, as these references will become
         * invalid.
         *
         * @param image image being analyzed VERY IMPORTANT: Analyzer method implementation must
         * call image.close() on received images when finished using them. Otherwise, new images
         * may not be received or the camera may stall, depending on back pressure setting.
         *
         */
        override fun analyze(image: ImageProxy) {
            // If there are no listeners attached, we don't need to perform analysis
            if (listeners.isEmpty()) {
                image.close()
                return
            }

            // Keep track of frames analyzed
            val currentTime = System.currentTimeMillis()
            frameTimestamps.push(currentTime)

            // Compute the FPS using a moving average
            while (frameTimestamps.size >= frameRateWindow) frameTimestamps.removeLast()
            val timestampFirst = frameTimestamps.peekFirst() ?: currentTime
            val timestampLast = frameTimestamps.peekLast() ?: currentTime
            framesPerSecond = 1.0 / ((timestampFirst - timestampLast) /
                    frameTimestamps.size.coerceAtLeast(1).toDouble()) * 1000.0

            // Analysis could take an arbitrarily long amount of time
            // Since we are running in a different thread, it won't stall other use cases

            lastAnalyzedTimestamp = frameTimestamps.first

            // Since format in ImageAnalysis is YUV, image.planes[0] contains the luminance plane
            val buffer = image.planes[0].buffer

            // Extract image data from callback object
            val data = buffer.toByteArray()

            // Convert the data into an array of pixel values ranging 0-255
            val pixels = data.map { it.toInt() and 0xFF }

            // Compute average luminance for the image
            val luma = pixels.average()

            // Call all listeners with new value
            listeners.forEach { it(luma) }

            image.close()
        }
    }

    fun enableTorch(enable: Boolean){
        isTorchEnable = enable
        camera?.cameraControl?.enableTorch(enable)
    }

    fun torchState(): LiveData<Int>? {
        return camera?.cameraInfo?.torchState
    }

    fun takePicture(){
        imageCapture?.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
            @SuppressLint("UnsafeOptInUsageError")
            override fun onCaptureSuccess(imageProxy: ImageProxy) {
                onPictureTaken(imageProxy)
            }
        })
    }

    fun enableAutoFocus(enable: Boolean){
        if(enable) {
            val factory: MeteringPointFactory = SurfaceOrientedMeteringPointFactory(
                    cameraPreview.width.toFloat(), cameraPreview.height.toFloat())
            val centerWidth = cameraPreview.width.toFloat() / 2
            val centerHeight = cameraPreview.height.toFloat() / 2
            //create a point on the center of the view
            val autoFocusPoint = factory.createPoint(centerWidth, centerHeight)
            try {
                camera?.cameraControl?.startFocusAndMetering(
                        FocusMeteringAction.Builder(
                                autoFocusPoint,
                                FocusMeteringAction.FLAG_AF
                        ).apply {
                            //auto-focus every 1 seconds
                            setAutoCancelDuration(1, TimeUnit.SECONDS)
                        }.build()
                )
            } catch (e: CameraInfoUnavailableException) {
                Log.d("ERROR", "cannot access camera", e)
            }
        } else {
            camera?.cameraControl?.cancelFocusAndMetering()
        }
        isAutoFocusEnable = enable
    }



    ////////////////////////////////////////////////////////////////////////////////////////
    //
    //        Permissions
    //
    ////////////////////////////////////////////////////////////////////////////////////////

    protected fun hasCameraPermission():Boolean{
        return ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    //
    //        Permissions
    //
    ////////////////////////////////////////////////////////////////////////////////////////

    fun onCallbackRequestPermissions(results :MutableMap<String, Boolean>) {
        if(!results.isEmpty()) {
            if(isAdded && isResumed) {
                checkPermissions(requestedPermissions)
            }
        }
    }

    fun onCallbackRequestActivitySettings(result :androidx.activity.result.ActivityResult) {
       // checkPermissions(requestedPermissions)
    }

    protected fun checkPermissions(permissions:ArrayList<String> = ArrayList()) {
        //request permission
        val hasPermissionCamera = ContextCompat.checkSelfPermission(requireContext(),
            Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (!hasPermissionCamera && !permissions.contains(Manifest.permission.CAMERA)) {
            permissions.add(Manifest.permission.CAMERA)
        }

        val checkPermissions = PermissionUtils.checkPermissions(requireActivity(), permissions)
        val permissionsGranted =
            PermissionUtils.getPermissions(checkPermissions, PermissionStatus.GRANTED)
        val permissionsRevoked =
            PermissionUtils.getPermissions(checkPermissions, PermissionStatus.REVOKED)
        val permissionsDenied =
            PermissionUtils.getPermissions(checkPermissions, PermissionStatus.DENIED)

        onPermissionResult(permissionsGranted, permissionsDenied, permissionsRevoked)

        if (PermissionUtils.allPermissionsGranted(checkPermissions)) {
            //Dont do anything
            cameraPreview.post {

                // Keep track of the display in which this view is attached
                displayId = cameraPreview.display.displayId

                // Set up the camera and its use cases
                setUpCamera()
            }
        } else {

            if (!permissionsRevoked.isEmpty()) {
                for(permission in permissionsRevoked) {
                    when (permission) {
                        Manifest.permission.CAMERA->{
                            if(isAutoCameraPermissionRevokedHandled) {
                                val builder = AlertDialog.Builder(context)
                                builder.setTitle(getString(R.string.permission_camera_title))
                                    .setMessage(R.string.permission_camera_rationale)
                                    .setPositiveButton(android.R.string.ok) { dialog, which ->
                                        if (isAdded && isResumed) {
                                            checkPermissions(requestedPermissions)
                                        }
                                        dialog.dismiss()
                                    }
                                    .setNegativeButton(R.string.button_go_back) { dialog, which ->
                                        if (isAdded && isResumed) {
                                            requireActivity().onBackPressed()
                                        }
                                        dialog.dismiss()
                                    }
                                    .setNeutralButton(R.string.button_go_settings) { dialog, which ->
                                        if (isAdded && isResumed) {
                                            //Go to settings
                                            registerActivitySettingsForResult?.launch(
                                                Intent(
                                                    Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                                    Uri.parse("package:" + getApplicationId())
                                                )
                                            )
                                        }
                                        dialog.dismiss()
                                    }
                                    .show()
                            }
                        }
                    }
                }
            } else {
                PermissionUtils.requestPermissionsDeniedToUser(requireContext(), permissionsDenied, permissionLauncher)
            }
        }
    }

    abstract fun onPermissionResult(permissionsGranted: ArrayList<String>, permissionsDenied: ArrayList<String>, permissionsRevoked: ArrayList<String>)
    abstract fun getApplicationId():String

    val rotation:Int
    get() {
        return getRotation(lensFacing)
    }

    private fun getRotation(lensPosition: Int = CameraSelector.LENS_FACING_BACK):Int{
        //return cameraPreview.display.rotation
        try {
            if (camera != null) {
                val sensorRotationDegrees = camera?.cameraInfo?.sensorRotationDegrees!!
                val rotation = cameraPreview.display.rotation

                var degrees = 0
                when (rotation) {
                    Surface.ROTATION_0 -> degrees = 0
                    Surface.ROTATION_90 -> degrees = 90
                    Surface.ROTATION_180 -> degrees = 180
                    Surface.ROTATION_270 -> degrees = 270
                }
                var result: Int
                if (lensPosition == CameraSelector.LENS_FACING_FRONT) {
                    result = (sensorRotationDegrees + degrees - 360) % 360
                    result = (360 + result) % 360  // compensate the mirror
                } else {  // back-facing
                    result = (sensorRotationDegrees - degrees + 360) % 360
                }
                return result
            } else {
                return cameraPreview.display.rotation
            }
        }catch (e:Exception){
            return 0
        }
    }


    /**
     * Shows an error message dialog.
     */
    class ErrorDialog : androidx.fragment.app.DialogFragment() {

        override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
            val activity = activity
            return AlertDialog.Builder(activity)
                    .setMessage(requireArguments().getString(ARG_MESSAGE))
                    .setPositiveButton(android.R.string.ok) { dialogInterface, i -> activity!!.finish() }
                    .create()
        }

        companion object {

            private val ARG_MESSAGE = "message"

            fun newInstance(message: String): ErrorDialog {
                val dialog = ErrorDialog()
                val args = Bundle()
                args.putString(ARG_MESSAGE, message)
                dialog.arguments = args
                return dialog
            }
        }

    }

    abstract fun onPictureTaken(image: ImageProxy)
    abstract fun onImageProxy(image: ImageProxy)

    fun switchCameras(){
        lensFacing = if (CameraSelector.LENS_FACING_FRONT == lensFacing) {
            CameraSelector.LENS_FACING_BACK
        } else {
            CameraSelector.LENS_FACING_FRONT
        }
        // Re-bind use cases to update selected camera
        bindCameraUseCases()
    }





    fun ImageProxy.toInputImage(): InputImage{
        val y = this.planes[0]
        val u = this.planes[1]
        val v = this.planes[2]

        val Yb = y.buffer.remaining()
        val Ub = u.buffer.remaining()
        val Vb = v.buffer.remaining()

        val data = ByteArray(Yb + Ub + Vb)

        y.buffer.get(data, 0, Yb)
        u.buffer.get(data, Yb, Ub)
        v.buffer.get(data, Yb + Ub, Vb)



        val inputImage = InputImage.fromByteArray(data,
                this.width,
                this.height,
                rotation,
                InputImage.IMAGE_FORMAT_YV12
        )

        return inputImage
    }

    fun ImageProxy.toByteArray(): ByteArray{
        val buffer = this.planes[0].buffer
        buffer.rewind()
        val byteArray = ByteArray(buffer.capacity())
        buffer.get(byteArray)
        val clone = byteArray.clone()
        //val nv21 = yuv420888ToNv21(this)
        return clone
    }

    fun ImageProxy.toBitmap(): Bitmap? {
        val toByteArray = this.toByteArray()
        return BitmapFactory.decodeByteArray(toByteArray, 0, toByteArray.size)
    }

    private fun YuvImage.toBitmap(): Bitmap? {
        val out = ByteArrayOutputStream()
        if (!compressToJpeg(Rect(0, 0, width, height), 100, out))
            return null
        val imageBytes: ByteArray = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val format = image.format
        val pixelCount = image.cropRect.width() * image.cropRect.height()
        val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
        val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
        imageToByteBuffer(image, outputBuffer, pixelCount)
        return outputBuffer
    }

    private fun imageToByteBuffer(image: ImageProxy, outputBuffer: ByteArray, pixelCount: Int) {
        assert(image.format == ImageFormat.YUV_420_888)

        val imageCrop = image.cropRect
        val imagePlanes = image.planes

        imagePlanes.forEachIndexed { planeIndex, plane ->
            // How many values are read in input for each output value written
            // Only the Y plane has a value for every pixel, U and V have half the resolution i.e.
            //
            // Y Plane            U Plane    V Plane
            // ===============    =======    =======
            // Y Y Y Y Y Y Y Y    U U U U    V V V V
            // Y Y Y Y Y Y Y Y    U U U U    V V V V
            // Y Y Y Y Y Y Y Y    U U U U    V V V V
            // Y Y Y Y Y Y Y Y    U U U U    V V V V
            // Y Y Y Y Y Y Y Y
            // Y Y Y Y Y Y Y Y
            // Y Y Y Y Y Y Y Y
            val outputStride: Int

            // The index in the output buffer the next value will be written at
            // For Y it's zero, for U and V we start at the end of Y and interleave them i.e.
            //
            // First chunk        Second chunk
            // ===============    ===============
            // Y Y Y Y Y Y Y Y    V U V U V U V U
            // Y Y Y Y Y Y Y Y    V U V U V U V U
            // Y Y Y Y Y Y Y Y    V U V U V U V U
            // Y Y Y Y Y Y Y Y    V U V U V U V U
            // Y Y Y Y Y Y Y Y
            // Y Y Y Y Y Y Y Y
            // Y Y Y Y Y Y Y Y
            var outputOffset: Int

            when (planeIndex) {
                0 -> {
                    outputStride = 1
                    outputOffset = 0
                }
                1 -> {
                    outputStride = 2
                    // For NV21 format, U is in odd-numbered indices
                    outputOffset = pixelCount + 1
                }
                2 -> {
                    outputStride = 2
                    // For NV21 format, V is in even-numbered indices
                    outputOffset = pixelCount
                }
                else -> {
                    // Image contains more than 3 planes, something strange is going on
                    return@forEachIndexed
                }
            }

            val planeBuffer = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride

            // We have to divide the width and height by two if it's not the Y plane
            val planeCrop = if (planeIndex == 0) {
                imageCrop
            } else {
                Rect(
                        imageCrop.left / 2,
                        imageCrop.top / 2,
                        imageCrop.right / 2,
                        imageCrop.bottom / 2
                )
            }

            val planeWidth = planeCrop.width()
            val planeHeight = planeCrop.height()

            // Intermediate buffer used to store the bytes of each row
            val rowBuffer = ByteArray(plane.rowStride)

            // Size of each row in bytes
            val rowLength = if (pixelStride == 1 && outputStride == 1) {
                planeWidth
            } else {
                // Take into account that the stride may include data from pixels other than this
                // particular plane and row, and that could be between pixels and not after every
                // pixel:
                //
                // |---- Pixel stride ----|                    Row ends here --> |
                // | Pixel 1 | Other Data | Pixel 2 | Other Data | ... | Pixel N |
                //
                // We need to get (N-1) * (pixel stride bytes) per row + 1 byte for the last pixel
                (planeWidth - 1) * pixelStride + 1
            }

            for (row in 0 until planeHeight) {
                // Move buffer position to the beginning of this row
                planeBuffer.position(
                        (row + planeCrop.top) * rowStride + planeCrop.left * pixelStride)

                if (pixelStride == 1 && outputStride == 1) {
                    // When there is a single stride value for pixel and output, we can just copy
                    // the entire row in a single step
                    planeBuffer.get(outputBuffer, outputOffset, rowLength)
                    outputOffset += rowLength
                } else {
                    // When either pixel or output have a stride > 1 we must copy pixel by pixel
                    planeBuffer.get(rowBuffer, 0, rowLength)
                    for (col in 0 until planeWidth) {
                        outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                        outputOffset += outputStride
                    }
                }
            }
        }


    }


    companion object {

        private val TAG = CameraXFragment::class.java.simpleName
        private const val RATIO_4_3_VALUE = 4.0 / 3.0
        private const val RATIO_16_9_VALUE = 16.0 / 9.0

        val KEY_AUTO_FOCUS = "KEY_AUTO_FOCUS"
        val KEY_IS_TORCH_ENABLE = "KEY_IS_TORCH_ENABLE"
        private val KEY_AUTO_HANDLE_CAMERA_PERMISSION_REVOKED = "KEY_AUTO_HANDLE_CAMERA_PERMISSION"

        /** Helper function used to create a timestamped file */
        private fun createFile(baseFolder: File, format: String, extension: String) =
                File(baseFolder, SimpleDateFormat(format, Locale.US)
                        .format(System.currentTimeMillis()) + extension)
    }
}
