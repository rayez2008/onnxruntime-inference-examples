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
import android.widget.FrameLayout
import androidx.core.app.ActivityCompat


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var previewView: PreviewView
    private var currentImageProxy: ImageProxy? = null
    private lateinit var classes: List<String>

    private val cameraExecutor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
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
        @OptIn(ExperimentalGetImage::class)
        override fun analyze(imageProxy: ImageProxy) {
            // Process the image only if currentImageProxy is null
            if (currentImageProxy == null) {
                currentImageProxy = imageProxy

                // Convert ImageProxy to InputStream for detection
                val inputStream = imageProxyToInputStream(imageProxy)

                // Perform object detection
                performObjectDetection(inputStream)

                // Close the imageProxy after processing
                imageProxy.close()
                currentImageProxy = null
            }
        }
    }

    private fun imageProxyToInputStream(imageProxy: ImageProxy): InputStream {
        // Implement the conversion logic from ImageProxy to InputStream
        val image = imageProxy.image ?: throw IllegalArgumentException("ImageProxy does not contain an image")
        val planes = image.planes
        val buffer = planes[0].buffer
        val byteArray = ByteArray(buffer.remaining())
        buffer.get(byteArray)
        return ByteArrayInputStream(byteArray)
    }

    private fun performObjectDetection(inputStream: InputStream) {
        try {
            // Assuming you have an ObjectDetector class for detection
            val objDetector = ObjectDetector()
            val result = objDetector.detect(inputStream, ortEnv, ortSession)
            updateUI(result)
        } catch (e: Exception) {
            Log.e(TAG, "Exception caught when performing object detection", e)
        }
    }

    private fun updateUI(result: Result) {
        // Create a mutable copy of the output bitmap for drawing
        val mutableBitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)

        // Create a canvas to draw on the mutable bitmap
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.color = Color.RED // Bounding box color
        paint.strokeWidth = 5f // Bounding box stroke width
        paint.style = Paint.Style.STROKE // Draw only the outline

        // Iterate over the output boxes and draw them on the canvas
        for (box in result.outputBox) {
            val left = box[0]
            val top = box[1]
            val right = box[0] + box[2]
            val bottom = box[1] + box[3]

            // Draw the bounding box
            canvas.drawRect(left, top, right, bottom, paint)

            // Draw the label on the bounding box
            paint.color = Color.WHITE // Text color
            paint.textSize = 28f // Text size
            val label = "Detected" // You can replace this with the actual class label
            canvas.drawText(label, left, top - 10, paint) // Draw label above the box
        }

        // Now update the overlay of the PreviewView
        // Clear previous overlays
        previewView.overlay.clear()

        // Create a new ImageView to display the processed bitmap
        val overlayView = ImageView(this)
        overlayView.setImageBitmap(mutableBitmap)

        // Make the overlay view match the preview view size
        val layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        overlayView.layoutParams = layoutParams

        // Add the overlay view to the PreviewView
        previewView.addView(overlayView)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.yolov8n_with_pre_post_processing
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    companion object {
        const val TAG = "ORTObjectDetection"
        const val CAMERA_PERMISSION_REQUEST_CODE = 1001
    }
}