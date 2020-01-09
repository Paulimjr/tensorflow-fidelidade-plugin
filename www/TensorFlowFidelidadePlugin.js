var exec = require('cordova/exec');

/**
 * Returns tensorflow value
 * 
 *  Use can see this example below:
 * 
 *      var imageBase64 = "IMG_BASE";
 *      var modelName = "example_model";
 * 
        cordova.plugins.TensorFlowFidelidadePlugin.loadModel(modelName, imageBase64,
            function success(data) {
                alert("Result: "+data.value);
            },
            function error(data) {
                alert("Error: "+data);
            }
        );
 * 
 * @param modelName the model name of the file already exists
 * @param imagePath the image path with image content
 * @param successCallback the success
 * @param errorCallback the error
 */
exports.executeModel = function(modelName, imagePath, successCallback, errorCallback) {
    exec(successCallback, errorCallback,  'TensorFlowFidelidadePlugin', 'executeModel', [modelName, imagePath]);
};

exports.loadModels = function(enqModel, qualityModel, unetModel, successCallback, errorCallback) {
    exec(successCallback, errorCallback,  'TensorFlowFidelidadePlugin', 'loadModels', [enqModel, qualityModel, unetModel]);
};