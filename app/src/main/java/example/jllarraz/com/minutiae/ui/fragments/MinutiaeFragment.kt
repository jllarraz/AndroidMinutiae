package example.jllarraz.com.minutiae.ui.fragments

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import example.jllarraz.com.myapplication.BuildConfig
import example.jllarraz.com.minutiae.utils.ImageProcessing
import example.jllarraz.com.minutiae.utils.ImageUtils
import example.jllarraz.com.minutiae.utils.ImageUtils.rotate
import example.jllarraz.com.myapplication.R
import example.jllarraz.com.myapplication.databinding.FragmentMinutiaeBinding
import io.reactivex.Single
import io.reactivex.android.schedulers.AndroidSchedulers

import io.reactivex.disposables.CompositeDisposable
import io.reactivex.schedulers.Schedulers
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean

class MinutiaeFragment : CameraXFragment() {



    private var callBack: CallBack? = null
    var disposable = CompositeDisposable()
    private var fragmentBinding: FragmentMinutiaeBinding?=null

    var processingImage:AtomicBoolean = AtomicBoolean(false)
    var openCvLoaded:AtomicBoolean = AtomicBoolean(false)
    private var baseLoaderCallback:BaseLoaderCallback?=null



    var onFingerprintProgressReceiver: BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctxt: Context?, intent: Intent?) {
            when(intent?.action){
                ImageProcessing.Companion.ACTION_STEP_START_FINGERPRINT_EXTRACTION->{
                    fragmentBinding?.progressCircular?.visibility = View.VISIBLE
                    fragmentBinding?.log?.setText(getString(R.string.step_start))
                }
                ImageProcessing.Companion.ACTION_STEP_SKIN_DETECTION->{
                    fragmentBinding?.log?.setText(getString(R.string.step_skin_detection))
                }
                ImageProcessing.Companion.ACTION_STEP_HISTOGRAM_EQUALIZATION->{
                    fragmentBinding?.log?.setText(getString(R.string.step_histogram_equalization))
                }
                ImageProcessing.Companion.ACTION_STEP_CLAHE_EQUALIZATION->{
                    fragmentBinding?.log?.setText(getString(R.string.step_clahe_equalization))
                }
                ImageProcessing.Companion.ACTION_STEP_ADAPTATIVE_THRESHOLD->{
                    fragmentBinding?.log?.setText(getString(R.string.step_adaptative_threshold))
                }
                ImageProcessing.Companion.ACTION_STEP_BILATERAL_FILTER->{
                    fragmentBinding?.log?.setText(getString(R.string.step_bilateral_filter))
                }
                ImageProcessing.Companion.ACTION_STEP_FINGERPRINT_SKELETIZATION->{
                    fragmentBinding?.log?.setText(getString(R.string.step_fingerprint_skelization))
                }
                ImageProcessing.Companion.ACTION_STEP_RIDGE_THINNING->{
                    fragmentBinding?.log?.setText(getString(R.string.step_ridge_thinning))
                }
                ImageProcessing.Companion.ACTION_STEP_MINUTIAE_EXTRATION->{
                    fragmentBinding?.log?.setText(getString(R.string.step_minutiae_extraction))
                }
                ImageProcessing.Companion.ACTION_STEP_FINISH_FINGERPRINT_EXTRACTION->{
                    fragmentBinding?.progressCircular?.visibility = View.GONE
                    fragmentBinding?.log?.setText("")
                }
            }
        }
    }




    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?,
                              savedInstanceState: Bundle?): View? {
        fragmentBinding = FragmentMinutiaeBinding.inflate(inflater, container, false)
        lensFacing = CameraSelector.LENS_FACING_BACK
        return fragmentBinding?.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        baseLoaderCallback = object : BaseLoaderCallback(requireContext()) {
            override fun onManagerConnected(status: Int) {
                when (status) {
                    LoaderCallbackInterface.SUCCESS -> {
                        Toast.makeText(requireContext(), "OpenCV SUCCESS", Toast.LENGTH_SHORT)
                            .show()
                        Log.d(TAG, "Opencv SUCCESS")
                        openCvLoaded.set(true)
                    }
                    LoaderCallbackInterface.INIT_FAILED -> {
                        Toast.makeText(requireContext(), "OpenCV INIT_FAILED", Toast.LENGTH_SHORT)
                            .show()
                        Log.d(TAG, "Opencv INIT_FAILED")
                        super.onManagerConnected(status)
                    }
                    LoaderCallbackInterface.INCOMPATIBLE_MANAGER_VERSION -> {
                        Toast.makeText(
                            requireContext(),
                            "OpenCV INCOMPATIBLE_MANAGER_VERSION",
                            Toast.LENGTH_SHORT
                        ).show()
                        Log.d(TAG, "Opencv INCOMPATIBLE_MANAGER_VERSION")
                        super.onManagerConnected(status)
                    }
                    LoaderCallbackInterface.INSTALL_CANCELED -> {
                        Toast.makeText(
                            requireContext(),
                            "OpenCV INSTALL_CANCELED",
                            Toast.LENGTH_SHORT
                        ).show()
                        Log.d(TAG, "Opencv INSTALL_CANCELED")
                        super.onManagerConnected(status)
                    }
                    LoaderCallbackInterface.MARKET_ERROR -> {
                        Toast.makeText(requireContext(), "OpenCV MARKET_ERROR", Toast.LENGTH_SHORT)
                            .show()
                        Log.d(TAG, "Opencv MARKET_ERROR")
                        super.onManagerConnected(status)
                    }
                }
            }
        }

        val intentFilter = IntentFilter()
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_FINGERPRINT_SKELETIZATION)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_FINISH_FINGERPRINT_EXTRACTION)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_HISTOGRAM_EQUALIZATION)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_CLAHE_EQUALIZATION)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_ADAPTATIVE_THRESHOLD)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_BILATERAL_FILTER)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_MINUTIAE_EXTRATION)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_RIDGE_THINNING)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_SKIN_DETECTION)
        intentFilter.addAction(ImageProcessing.Companion.ACTION_STEP_START_FINGERPRINT_EXTRACTION)
        requireContext().registerReceiver(
            onFingerprintProgressReceiver,
            intentFilter
        )


        fragmentBinding?.buttonTakePicture?.setOnClickListener {
            takePicture()
        }

        fragmentBinding?.autoFocus?.setOnClickListener {
            enableAutoFocus(!isAutoFocusEnable)
            fragmentBinding?.autoFocus?.setText(if(isAutoFocusEnable) getString(R.string.disable_autofocus) else getString(R.string.enable_autofocus))
        }

        fragmentBinding?.layoutMinutiae?.buttonBackMinutiae?.setOnClickListener {
            fragmentBinding?.layoutButtons?.visibility = View.VISIBLE
            fragmentBinding?.layoutMinutiae?.root?.visibility = View.GONE
        }

        if(savedInstanceState == null){
            //Enable Autofocus by default, to focus the center of the screen and get a better finger image
            enableAutoFocus(true)
        }
    }

    override fun onAttach(context: Context) {
        super.onAttach(context)
        val activity = activity
        if (activity is CallBack) {
            callBack = activity
        }
    }

    override fun onDetach() {
        callBack = null
        super.onDetach()
    }


    override fun onResume() {
        fragmentBinding?.autoFocus?.setText(if(isAutoFocusEnable) "Disable Focus" else "Enable Focus")

        if(!OpenCVLoader.initDebug()){
            Log.e(TAG, "Unable to load OpenCV!")
            OpenCVLoader.initAsync(
                OpenCVLoader.OPENCV_VERSION,
                requireContext(),
                baseLoaderCallback
            )

        }else{
            Log.d(TAG, "OpenCV loaded Successfully!")
            baseLoaderCallback?.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }

        super.onResume()
    }

    override fun onPause() {
        super.onPause()
    }

    override fun onDestroyView() {
        if (!disposable.isDisposed()) {
            disposable.dispose();
        }
        fragmentBinding = null
        requireContext().unregisterReceiver(onFingerprintProgressReceiver)
        super.onDestroyView()
    }


    ////////////////////////////////////////////////////////////////////////////////////////
    //
    //        Events from camera2 fragment
    //
    ////////////////////////////////////////////////////////////////////////////////////////

    override val cameraPreview: PreviewView
        get(){
            return fragmentBinding?.cameraPreview!!
        }

    override val requestedPermissions: ArrayList<String>
        get() {
            return ArrayList<String>()
        }

    override fun onPermissionResult(
        permissionsGranted: ArrayList<String>,
        permissionsDenied: ArrayList<String>,
        permissionsRevoked: ArrayList<String>
    ) {

    }

    override fun getApplicationId(): String {
        return BuildConfig.APPLICATION_ID
    }

    override fun onPictureTaken(image: ImageProxy) {
        if(!processingImage.get()) {
            val subscribe = Single.fromCallable {
                val toByteArray = image.toByteArray()
                image.close()
                toByteArray
            }.flatMap{ byteArray->
                val processedImage = ImageProcessing().processedImage(
                    data = byteArray,
                    context = requireContext(),
                    isClahe = true,
                    isAdaptativeThreshold = true,
                    isBilateralFilter = true,
                )
                processedImage
            }
                .subscribeOn(Schedulers.computation())
                .doOnSubscribe {
                    processingImage.set(true)
                }
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(
                    { imageMinutiae ->
                        fragmentBinding?.layoutMinutiae?.imageMinutiae?.setImageBitmap(imageMinutiae)
                        fragmentBinding?.layoutMinutiae?.root?.visibility = View.VISIBLE
                        fragmentBinding?.layoutButtons?.visibility = View.GONE
                        processingImage.set(false)
                    }, { error ->
                        fragmentBinding?.layoutButtons?.visibility = View.VISIBLE
                        fragmentBinding?.layoutMinutiae?.root?.visibility = View.GONE
                        processingImage.set(false)
                    }
                )
            disposable.add(subscribe)
        }else {
            image.close()
        }
    }

    override fun onImageProxy(image: ImageProxy) {
        image.close()
    }


    ////////////////////////////////////////////////////////////////////////////////////////
    //
    //        Listener
    //
    ////////////////////////////////////////////////////////////////////////////////////////

    interface CallBack {
        fun onError()
    }

    companion object {
        private val TAG = MinutiaeFragment::class.java.simpleName

        fun newInstance(): MinutiaeFragment {
            return MinutiaeFragment()
        }
    }
}
