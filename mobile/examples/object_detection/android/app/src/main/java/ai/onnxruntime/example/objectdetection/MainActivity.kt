package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Matrix
import android.widget.FrameLayout
import androidx.core.app.ActivityCompat
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var previewView: PreviewView
    private lateinit var outputImage: ImageView
    private var currentImageProxy: ImageProxy? = null
    private lateinit var classes: List<String>

    private val cameraExecutor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        outputImage = findViewById(R.id.imageView1)
        classes = readClasses()

        // Initialize Ort Session
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        // Check camera permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
        } else {
            startCamera() // Start camera if permission is already granted
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
        currentImageProxy?.close() // Close current image if still available
        cameraExecutor.shutdown() // Shutdown the executor
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (currentImageProxy == null) {
                currentImageProxy = imageProxy

                val jpegBytes = imageProxyToJpegByteArray(imageProxy)
                val inputStream = ByteArrayInputStream(jpegBytes)

                performObjectDetection(inputStream)

                imageProxy.close()
                currentImageProxy = null
            }
        }
    }

    private fun imageProxyToJpegByteArray(imageProxy: ImageProxy): ByteArray {
        val yuvImage = yuvToJpeg(imageProxy)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, outputStream)
        return outputStream.toByteArray()
    }

    private fun imageProxyToInputStream(imageProxy: ImageProxy): InputStream {
        // Convert ImageProxy (YUV) to JPEG format
        val yuvImage = yuvToJpeg(imageProxy)

        // Convert YUV image to InputStream (JPEG format)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, yuvImage.width, yuvImage.height), 100, outputStream)

        // Return the resulting InputStream
        return ByteArrayInputStream(outputStream.toByteArray())
    }

    // Convert YUV ImageProxy to JPEG using YuvImage
    private fun yuvToJpeg(imageProxy: ImageProxy): YuvImage {
        val image = imageProxy.image ?: throw IllegalArgumentException("ImageProxy does not contain a valid image")

        val yBuffer: ByteBuffer = image.planes[0].buffer // Y
        val uBuffer: ByteBuffer = image.planes[1].buffer // U
        val vBuffer: ByteBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copy Y buffer to NV21 array
        yBuffer.get(nv21, 0, ySize)

        // Copy U and V buffer to NV21 array
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        // Create YUV image from NV21 data and compress it to JPEG
        return YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
    }

    private fun updateUIWithDetectionResults(result: Result) {
        // Rotate the output bitmap by 90 degrees
        val matrix = Matrix()
        matrix.postRotate(90f)
        val rotatedBitmap = Bitmap.createBitmap(result.outputBitmap, 0, 0, result.outputBitmap.width, result.outputBitmap.height, matrix, true)

        // Create a mutable copy of the rotated bitmap for drawing
        val mutableBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)

        // Create a canvas to draw on the mutable bitmap
        val canvas = Canvas(mutableBitmap)

        // Rotate the canvas to align bounding boxes with the image
        canvas.rotate(90f, (mutableBitmap.width / 2).toFloat(), (mutableBitmap.height / 2).toFloat())

        val paint = Paint()
        paint.color = Color.RED // Bounding box color
        paint.strokeWidth = 5f // Bounding box stroke width
        paint.style = Paint.Style.STROKE // Draw only the outline
        paint.textSize = 28f // Text size
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER) // Text overlapping pattern

        // Draw bounding boxes and labels
        result.outputBox.forEach { box_info ->
            val left = box_info[0] - box_info[2] / 2
            val top = box_info[1] - box_info[3] / 2
            val right = box_info[0] + box_info[2] / 2
            val bottom = box_info[1] + box_info[3] / 2

            // Draw the bounding box
            canvas.drawRect(left, top, right, bottom, paint)

            // Draw the label
            canvas.drawText("%s:%.2f".format(classes[box_info[5].toInt()], box_info[4]), left, top - 10, paint)
        }

        // Set the rotated and annotated bitmap to the ImageView
        outputImage.setImageBitmap(mutableBitmap)
    }


    private fun readModel(): ByteArray {
        val modelID = R.raw.yolov8n_with_pre_post_processing
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    private fun performObjectDetection(inputStream: InputStream) {
        try {
            val objDetector = ObjectDetector()
            val result = objDetector.detect(inputStream, ortEnv, ortSession)
            updateUIWithDetectionResults(result)
//            runOnUiThread {
//                updateUIWithDetectionResults(result)
//            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during object detection", e)
//            runOnUiThread {
//                Toast.makeText(this, "${e.cause}, ${e.message}", Toast.LENGTH_SHORT).show()
//            }
        }
    }

//    fun performObjectDetection(inputStream: InputStream) {
//        var objDetector = ObjectDetector()
//        inputStream.reset()
//        var result = objDetector.detect(inputStream, ortEnv, ortSession)
//        updateUIWithDetectionResults(result);
//    }

    companion object {
        const val TAG = "ORTObjectDetection"
        const val CAMERA_PERMISSION_REQUEST_CODE = 1001
    }
}