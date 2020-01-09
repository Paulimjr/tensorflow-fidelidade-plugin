package com.tensorflow.fidelidade.plugin;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.util.Log;

import com.tensorflow.fidelidade.plugin.sources.TIOModel;
import com.tensorflow.fidelidade.plugin.sources.TIOModelBundle;
import com.tensorflow.fidelidade.plugin.sources.TIOModelBundleManager;
import com.tensorflow.fidelidade.plugin.sources.TIOModelException;
import com.tensorflow.fidelidade.plugin.sources.TIOVectorLayerDescription;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaInterface;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.PluginResult;
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
    // private TIOModel model;
    //Models (Enquadramento, Qualidade da Imagem e Verificar se esta cortada)
    private static final String ENQ_MODEL = "enq_model";
    private static final String QUALITY_MODEL = "quality_model";
    private static final String UNET_VEHICLE_MODEL = "unet_vehicle_model";
    private static final String ACTION_EXECUTE_MODEL = "executeModel";
    private CallbackContext callbackContext;
    private static String BAD_IMAGE = "BAD_IMAGE";
    private static String GOOD_IMAGE = "GOOD_IMAGE";
    private static final String ACTION_LOAD_MODELS = "loadModels";

    // Enq
    private TIOModel enqModel;

    // Quality
    private TIOModel qualityModel;

    // Unet
    private TIOModel unetModel;

    @Override
    public boolean execute(String action, JSONArray args, final CallbackContext callbackContext) throws JSONException {
        this.callbackContext = callbackContext;

        if (action != null && action.equalsIgnoreCase(ACTION_EXECUTE_MODEL)) {

            if (args != null && args.length() > 0) {

                String modelName = args.getString(0);
                String imageBase64 = args.getString(1);

                if (modelName != null && imageBase64 != null) {
                    this.executeModel(modelName, imageBase64);
                } else {
                    this.callbackContext.error("Invalid or not found action!");
                }

            } else {
                this.callbackContext.error("The arguments can not be null!");
            }

        } else if (action != null && action.equalsIgnoreCase(ACTION_LOAD_MODELS)) {
            this.loadModels(args.getString(0), args.getString(1), args.getString(2));
        } else if (action == null) {
            this.callbackContext.error("Invalid or not found action!");
        }
        return true;
    }

    public void loadModels(String enqModel, String qualityModel, String unetModel) {
        try {
            if (enqModel == null || qualityModel == null || unetModel == null) {
                this.callbackContext.error("You need to pass all models to load.");
            } else {
                TIOModelBundleManager manager = new TIOModelBundleManager(this.cordova.getActivity().getApplicationContext(), "");

                // Load enq Model
                TIOModelBundle enqModelBundle = manager.bundleWithId(enqModel);
                if (enqModelBundle == null) {
                    this.callbackContext.error("Model " + enqModel + " not found!");
                    return;
                } else {
                    this.enqModel = enqModelBundle.newModel();
                    this.enqModel.load();
                }

                //Load quality Model
                TIOModelBundle qualityModelBundle = manager.bundleWithId(qualityModel);
                if (qualityModelBundle == null) {
                    this.callbackContext.error("Model " + qualityModel + " not found!");
                    return;
                } else {
                    this.qualityModel = qualityModelBundle.newModel();
                    this.qualityModel.load();
                }

                //Load Unet Model
                TIOModelBundle unetModelBundle = manager.bundleWithId(unetModel);
                if (unetModelBundle == null) {
                    this.callbackContext.error("Model " + unetModel + " not found!");
                    return;
                } else {
                    this.unetModel = unetModelBundle.newModel();
                    this.unetModel.load();
                }

                PluginResult pluginResult = new PluginResult(PluginResult.Status.NO_RESULT);
                pluginResult.setKeepCallback(true);
                this.callbackContext.sendPluginResult(pluginResult);
            }
        } catch (Exception e) {
            this.callbackContext.error("Error to load models");
        }
    }

    /**
     * Load model to Tensor Flow Lite to execute a function
     */
    private synchronized void executeModel(String modelName, String imageBase64) {
        try {
            //Convert base64 to bitmap image
            Bitmap image = this.convertBase64ToBitmap(imageBase64);
            Bitmap imageResized;

            // Switch to check what is the model will be executed.
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
                    this.executeUnetVehicleModel(imageResized);
                    break;
                }
            }

        } catch (Exception e) {
            this.callbackContext.error("Error to load a model with name " + modelName);
        }
    }

    private synchronized Bitmap convertBase64ToBitmap(String base64String) {
        byte[] imageAsBytes = Base64.decode(base64String.getBytes(), Base64.DEFAULT);
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

    private synchronized void executeUnetVehicleModel(Bitmap imageResized) {
        cordova.getActivity().runOnUiThread(() -> {
            try {
                float[] result;
                result = (float[]) unetModel.runOn(imageResized);
                this.checkImage(result);
            } catch (Exception e) {
                callbackContext.error("Error to load or execute the Unet Vehicle model");
            }
        });
    }

    private synchronized void executeQualityModel(Bitmap imageResized) {
        cordova.getActivity().runOnUiThread(() -> {
            // Run the model on the input
            float[] result;

            try {
                result = (float[]) qualityModel.runOn(imageResized);
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

    private synchronized void executeFrameworkModel(Bitmap imageResized) {
        cordova.getActivity().runOnUiThread(() -> {
            // Run the model on the input
            float[] result = new float[0];

            try {
                result = (float[]) enqModel.runOn(imageResized);
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
                String[] labels = ((TIOVectorLayerDescription) enqModel.descriptionOfOutputAtIndex(0)).getLabels();

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

    private synchronized void checkImage(float[] data) {
        cordova.getActivity().runOnUiThread(() -> {

            //Arrays vertical horizontal
            float[] horizontalBorder1 = Arrays.copyOfRange(data, 0, 224);
            float[] horizontalBorder2 = Arrays.copyOfRange(data, 224, 448);
            float[] horizontalBorder3 = Arrays.copyOfRange(data, 448, 672);
            ////////////////////////////////////////////////////////////////////////
            float[] horizontalBorder4 = Arrays.copyOfRange(data, 49503, 49727);
            float[] horizontalBorder5 = Arrays.copyOfRange(data, 49727, 49951);
            float[] horizontalBorder6 = Arrays.copyOfRange(data, 49951, 50175);

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

            //Check HORIZONTAL COLUMNS
            if (checkLines(horizontalBorder1) || checkLines(horizontalBorder2)
                    || checkLines(horizontalBorder3)
                    || checkLines(horizontalBorder4)
                    || checkLines(horizontalBorder5)
                    || checkLines(horizontalBorder6)) {
                callbackContext.success(BAD_IMAGE);
            }

            //Check VERTICAL COLUMNS
            if (checkColumns(verticalBorder1) || checkColumns(verticalBorder2)
                    || checkColumns(verticalBorder3)
                    || checkColumns(verticalBorder4)
                    || checkColumns(verticalBorder5)
                    || checkColumns(verticalBorder6)) {
                callbackContext.success(BAD_IMAGE);
            } else {
                callbackContext.success(GOOD_IMAGE);
            }
        });
    }

    private synchronized boolean checkLines(float[] values) {
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

    private synchronized boolean checkColumns(ArrayList<Float> values) {
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