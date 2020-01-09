package com.tensorflow.fidelidade.plugin;

import android.annotation.TargetApi;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.util.Base64;
import android.util.Log;

import com.tensorflow.fidelidade.plugin.sources.TIOModel;
import com.tensorflow.fidelidade.plugin.sources.TIOModelBundle;
import com.tensorflow.fidelidade.plugin.sources.TIOModelBundleManager;
import com.tensorflow.fidelidade.plugin.sources.TIOModelException;
import com.tensorflow.fidelidade.plugin.sources.TIOVectorLayerDescription;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaPlugin;
import org.json.JSONArray;
import org.json.JSONException;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.PriorityQueue;


/**
 * OutSystems Experts Team
 * <p>
 * Author: Paulo Cesar
 * Date: 18-12-2019
 */
public class TensorFlowFidelidadePlugin extends CordovaPlugin {

    private static final String TAG = TensorFlowFidelidadePlugin.class.getSimpleName();
    private TIOModel model;
    //Models (Enquadramento, Qualidade da Imagem)
    private static final String ENQ_MODEL = "enq_model";
    private static final String QUALITY_MODEL = "quality_model";
    private static final String UNET_VEHICLE_MODEL = "unet_vehicle_model";
    private static final String ACTION_LOAD_MODEL = "loadModel";

    static final String ENQ_KEY = "enquadramento";

    // Handler to execute in Second Thread
    // Create a background thread
    private CallbackContext callbackContext;
    private static String BAD_IMAGE = "BAD_IMAGE";
    private static String GOOD_IMAGE = "GOOD_IMAGE";

    @Override
    public boolean execute(String action, JSONArray args, final CallbackContext callbackContext) throws JSONException {
        this.callbackContext = callbackContext;

        if (action != null && action.equalsIgnoreCase(ACTION_LOAD_MODEL)) {

            if (args != null && args.length() > 0) {

                String modelName = args.getString(0);
                String imageBase64 = args.getString(1);

                if (modelName != null && imageBase64 != null) {
                    this.loadModel(modelName, imageBase64);
                } else {
                    this.callbackContext.error("Invalid or not found action!");
                }

            } else {
                this.callbackContext.error("The arguments can not be null!");
            }

        } else {
            this.callbackContext.error("Invalid or not found action!");
        }

        return true;

    }

    /**
     * Load model to Tensor Flow Lite to execute a function
     */
    private void loadModel(String modelName, String imageBase64) {
        try {

            TIOModelBundleManager manager = new TIOModelBundleManager(this.cordova.getActivity().getApplicationContext(), "");
            // load the model
            TIOModelBundle bundle = manager.bundleWithId(modelName);

            if (bundle == null) {
                this.callbackContext.error("Model not found!");
                return;
            }

            model = bundle.newModel();
            model.load();

            //Convert base64 to bitmap image
            Bitmap image = this.convertBase64ToBitmap(imageBase64);

            // Model loaded success -- Resize Image
            Bitmap imageResized;


            // Switch to know what is the model will be executed.
            switch (modelName) {
                case ENQ_MODEL: {
                    imageResized = this.resizeImage(image, 64);
                    this.executeFrameworkModel(imageResized);
                    break;
                }

                case QUALITY_MODEL: {
                    imageResized = this.resizeImage(image, 224);
                    this.executeQualityModel(imageResized);
                    break;
                }

                case UNET_VEHICLE_MODEL: {
                    imageResized = this.resizeImage(image, 224);

                    // Start show Image
                  // Intent intent = new Intent(this.cordova.getActivity(), ImageTest.class);
                 //   intent.putExtra("img", imageResized);
                //    this.cordova.getActivity().startActivity(intent);

                    //sizeOf(imageResized);


                    this.executeUnetVehicleModel(imageResized);
                    break;
                }
            }

        } catch (Exception e) {
            this.callbackContext.error("Error to load a model with name " + modelName);
        }

    }

    @TargetApi(Build.VERSION_CODES.HONEYCOMB_MR1)
    protected int sizeOf(Bitmap data) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.HONEYCOMB_MR1) {
            return data.getRowBytes() * data.getHeight();
        } else {
            return data.getByteCount();
        }
    }

    private Bitmap convertBase64ToBitmap(String b64) {
        byte[] imageAsBytes = Base64.decode(b64.getBytes(), Base64.DEFAULT);
        return BitmapFactory.decodeByteArray(imageAsBytes, 0, imageAsBytes.length);
    }

    private synchronized Bitmap resizeImage(Bitmap img, int resizeImage) {

        try {
            if (img != null) {
                return Bitmap.createScaledBitmap(img, resizeImage, resizeImage, false);

            } else {
                this.callbackContext.error("Error to resize the image!");
            }
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            this.callbackContext.error(e.getMessage());
        }

        return null;
    }

    private void executeUnetVehicleModel(Bitmap imageResized) {
        this.cordova.getThreadPool().execute(() -> {
            try {
                float[] result;
                result = (float[]) model.runOn(imageResized);
                this.checkImage(result);

            } catch (Exception e) {
                callbackContext.error("Error to load or execute the Unet Vehicle model");
            }
        });
    }

    private void executeQualityModel(Bitmap imageResized) {
        this.cordova.getThreadPool().execute(() -> {
            // Run the model on the input
            float[] result;

            try {
                result = (float[]) model.runOn(imageResized);
                if (result.length > 0) {
                    if (result[0] > result[1]) {
                        callbackContext.success(String.valueOf(false));
                    } else {
                        callbackContext.success(String.valueOf(true));
                    }
                }

            } catch (Exception e) {
                callbackContext.error("Error to load or execute the quality model");
            }
        });
    }

    private void executeFrameworkModel(Bitmap imageResized) {
        this.cordova.getThreadPool().execute(() -> {
            // Run the model on the input
            float[] result = new float[0];

            try {
                result = (float[]) model.runOn(imageResized);
            } catch (TIOModelException e) {
                callbackContext.error("Error to execute the framework model");
            }

            // Build a PriorityQueue of the predictions
            PriorityQueue<Map.Entry<Integer, Float>> pq = new PriorityQueue<>(10, (o1, o2) -> (o2.getValue()).compareTo(o1.getValue()));
            for (int i = 0; i < 13; i++) {
                pq.add(new AbstractMap.SimpleEntry<>(i, result[i]));
            }

            try {
                // Show the 10 most likely predictions
                String[] labels = ((TIOVectorLayerDescription) model.descriptionOfOutputAtIndex(0)).getLabels();

                for (int i = 0; i < 1; i++) {

                    Map.Entry<Integer, Float> e = pq.poll();

                    if (e != null) {
                        callbackContext.success(labels[e.getKey()]);
                    }
                }

            } catch (Exception e) {
                callbackContext.error("Error to load or execute the framework model");
            }
        });
    }

    private void checkImage(float[] data) {

        this.cordova.getThreadPool().execute(() -> {
            //Arrays vertical horizontal
            float[] horizontalBorder1 = Arrays.copyOfRange(data, 0, 224);
            float[] horizontalBorder2 = Arrays.copyOfRange(data, 224, 448);
            float[] horizontalBorder3 = Arrays.copyOfRange(data, 448, 672);
            ////////////////////////////////////////////////////////////////////////
            float[] horizontalBorder4 = Arrays.copyOfRange(data, 49503, 49727);
            float[] horizontalBorder5 = Arrays.copyOfRange(data, 49727, 49951);
            float[] horizontalBorder6 = Arrays.copyOfRange(data, 49951, 50175);

            //Check HORIZONTAL
            if (checkLines(horizontalBorder1)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkLines(horizontalBorder2)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkLines(horizontalBorder3)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkLines(horizontalBorder4)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkLines(horizontalBorder5)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkLines(horizontalBorder6)) {
                callbackContext.success(BAD_IMAGE);
            }

            //Arrays vertical borders
            ArrayList<Float> verticalBorder1 = new ArrayList<>();
            ArrayList<Float> verticalBorder2 = new ArrayList<>();
            ArrayList<Float> verticalBorder3 = new ArrayList<>();
            ArrayList<Float> verticalBorder4 = new ArrayList<>();
            ArrayList<Float> verticalBorder5 = new ArrayList<>();
            ArrayList<Float> verticalBorder6 = new ArrayList<>();

            for (int index = 0; index < 224; index++) {
                float[] line = Arrays.copyOfRange(data, index * 224, (index + 1) * 224);

                verticalBorder1.add(line[0]);
                verticalBorder2.add(line[1]);
                verticalBorder3.add(line[2]);

                verticalBorder4.add(line[221]);
                verticalBorder5.add(line[222]);
                verticalBorder6.add(line[223]);
            }

            //Check VERTICAL COLUMNS
            if (checkColumns(verticalBorder1)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkColumns(verticalBorder2)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkColumns(verticalBorder3)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkColumns(verticalBorder4)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkColumns(verticalBorder5)) {
                callbackContext.success(BAD_IMAGE);
            } else if (checkColumns(verticalBorder6)) {
                callbackContext.success(BAD_IMAGE);
            } else {
                callbackContext.success(GOOD_IMAGE);
            }
        });
    }

    private boolean checkLines(float[] values) {

        int pixelLines = 0;

        for (int i = 0; i < 224; i++) {

            float currentValueAt = values[i];

            if (currentValueAt >= 0.5) {
                pixelLines++;
            } else {
                pixelLines = 0;
            }

            if (pixelLines == 10) {
                return true;
            }
        }
        return false;
    }

    private boolean checkColumns(ArrayList<Float> values) {
        int pixelLines = 0;

        for (int i = 0; i < 224; i++) {

            float currentValueAt = values.get(i);

            if (currentValueAt >= 0.5) {
                pixelLines++;
            } else {
                pixelLines = 0;
            }

            if (pixelLines == 10) {
                return true;
            }
        }
        return false;
    }
}