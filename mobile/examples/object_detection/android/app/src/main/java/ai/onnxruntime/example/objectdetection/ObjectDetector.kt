package ai.onnxruntime.example.objectdetection

//import ai.onnxruntime.OnnxJavaType
//import ai.onnxruntime.OrtSession
//import ai.onnxruntime.OnnxTensor
//import ai.onnxruntime.OrtEnvironment
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import java.io.InputStream
//import java.nio.ByteBuffer
//import java.util.*
//
//internal data class Result(
//    var outputBitmap: Bitmap,
//    var outputBox: Array<FloatArray>
//) {}
//
//internal class ObjectDetector(
//) {
//
//    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
//        // Step 1: convert image into byte array (raw image bytes)
//        val rawImageBytes = inputStream.readBytes()
//
//        // Step 2: get the shape of the byte array and make ort tensor
//        val shape = longArrayOf(rawImageBytes.size.toLong())
//
//        val inputTensor = OnnxTensor.createTensor(
//            ortEnv,
//            ByteBuffer.wrap(rawImageBytes),
//            shape,
//            OnnxJavaType.UINT8
//        )
//        inputTensor.use {
//            // Step 3: call ort inferenceSession run
//            val output = ortSession.run(Collections.singletonMap("image", inputTensor),
//                setOf("image_out","scaled_box_out_next")
//            )
//
//            // Step 4: output analysis
//            output.use {
//                val rawOutput = (output?.get(0)?.value) as ByteArray
//                val boxOutput = (output?.get(1)?.value) as Array<FloatArray>
//                val outputImageBitmap = byteArrayToBitmap(rawOutput)
//
//                // Step 5: set output result
//                var result = Result(outputImageBitmap,boxOutput)
//                return result
//            }
//        }
//    }
//
//    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
//        return BitmapFactory.decodeByteArray(data, 0, data.size)
//    }
//}

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.util.Collections

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
)

internal class ObjectDetector {
    companion object {
        private const val TAG = "ObjectDetector"
        private const val TARGET_WIDTH = 640
        private const val TARGET_HEIGHT = 640
    }

    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        try {
            // Step 1: convert image into byte array (raw image bytes)
//            val rawImageBytes = inputStream.readBytes()
            val rawImageBytes = resizeJpegIfNeeded(inputStream)
            Log.d(TAG, "Input image size: ${rawImageBytes.size} bytes")

            // Step 2: get the shape of the byte array and make ort tensor
            val shape = longArrayOf(rawImageBytes.size.toLong())
            Log.d(TAG, "Input tensor shape: ${shape.contentToString()}")

            val inputTensor = OnnxTensor.createTensor(
                ortEnv,
                ByteBuffer.wrap(rawImageBytes),
                shape,
                OnnxJavaType.UINT8
            )
            inputTensor.use {
                // Step 3: call ort inferenceSession run
                Log.d(TAG, "Running inference...")
                val output = ortSession.run(Collections.singletonMap("image", inputTensor),
                    setOf("image_out","scaled_box_out_next")
                )

                // Step 4: output analysis
                output.use {
                    Log.d(TAG, "Processing output...")
                    val rawOutput = (output?.get(0)?.value) as ByteArray
                    val boxOutput = (output?.get(1)?.value) as Array<FloatArray>
                    Log.d(TAG, "Raw output size: ${rawOutput.size} bytes")
                    Log.d(TAG, "Box output size: ${boxOutput.size} x ${boxOutput[0].size}")

                    val outputImageBitmap = byteArrayToBitmap(rawOutput)

                    // Step 5: set output result
                    var result = Result(outputImageBitmap, boxOutput)
                    return result
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during object detection", e)
            throw RuntimeException("${e.message}")
        }
    }

    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }

    private fun resizeJpegIfNeeded(inputStream: InputStream): ByteArray {
        val originalBitmap = BitmapFactory.decodeStream(inputStream)
        Log.d(TAG, "Original image size: ${originalBitmap.width} x ${originalBitmap.height}")

        if (originalBitmap.width != TARGET_WIDTH || originalBitmap.height != TARGET_HEIGHT) {
            val resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, TARGET_WIDTH, TARGET_HEIGHT, true)
            val outputStream = ByteArrayOutputStream()
            resizedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            Log.d(TAG, "Resized image to: $TARGET_WIDTH x $TARGET_HEIGHT")
            return outputStream.toByteArray()
        }

        // If no resizing is needed, return the original image bytes
        inputStream.reset()
        return inputStream.readBytes()
    }
}