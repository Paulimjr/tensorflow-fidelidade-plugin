//
//  TensorFlowFidelidadePlugin.mm
//
//  Created by Andre Grillo on 15/12/2019.
//  Copyright © 2019 Andre Grillo. All rights reserved.
//

#import <TensorIO/TensorIO-umbrella.h>
#import <Cordova/CDV.h>

@interface TensorFlowFidelidadePlugin: CDVPlugin {
    id model;
    CDVPluginResult *pluginResult;
}

@property (strong, nonatomic) CDVInvokedUrlCommand* commandHelper;
- (void)loadModel:(CDVInvokedUrlCommand*)command;

@end

@implementation TensorFlowFidelidadePlugin

- (void)loadModel:(CDVInvokedUrlCommand*)command {
    
    self.commandHelper = command;
    UIImage *inputImage = [self decodeBase64ToImage:[command.arguments objectAtIndex:1]];
    [command.arguments objectAtIndex:1];
    NSString *modelName = [command.arguments objectAtIndex:0];
    NSString *path = [NSBundle.mainBundle bundlePath];
    NSError *error;
    
    // Checks the model index on ModelBundleManager Array
    int modelIndex = 3;
    [TIOModelBundleManager.sharedManager loadModelBundlesAtPath:path error:&error];
    for (int i = 0; i < 3; i++) {
        if ([modelName isEqualToString:TIOModelBundleManager.sharedManager.modelBundles[i].identifier]) {
            modelIndex = i;
        }
    }
    
    //Checks if the modelIndex was found
    if (modelIndex > 2) {
        NSLog(@"Error: Model not found.");
        return;
    }
    
    TIOModelBundle *bundle = TIOModelBundleManager.sharedManager.modelBundles[modelIndex];
    model = [bundle newModel];
    
    UIImage *resizedImage;
    
    if ([modelName isEqualToString:@"enq_model"]){
        if (inputImage.size.width == 64.0 && inputImage.size.height == 64.0) {
            [self runModelEnquadramento:inputImage];
        } else {
            CGSize modelImageSize = CGSizeMake(64.0, 64.0);
            resizedImage = [self resizeImage:inputImage tosize:(modelImageSize)];
            [self runModelEnquadramento:resizedImage];
        }
    }
    else if ([modelName isEqualToString:@"quality_model"]){
        if (inputImage.size.width == 64.0 && inputImage.size.height == 64.0) {
            [self runModelEnquadramento:inputImage];
        } else {
            CGSize modelImageSize = CGSizeMake(224.0, 224.0);
            resizedImage = [self resizeImage:inputImage tosize:(modelImageSize)];
            [self runModelQuality:resizedImage];
        }
    }
    else if ([modelName isEqualToString:@"unet_vehicle_model"]){
        //if (image.size.width == 224.0 && image.size.height == 224.0) {
        //    [self runModelunet_vehicle_model:resizedImage];
        //} else {
            CGSize modelImageSize = CGSizeMake(224.0, 224.0);
            resizedImage = [self resizeImage:inputImage tosize:(modelImageSize)];
            [self runModelunet_vehicle_model:resizedImage];
        //}
    }
}

- (UIImage *)decodeBase64ToImage:(NSString *)strEncodeData {
  NSData *data = [[NSData alloc]initWithBase64EncodedString:strEncodeData options:NSDataBase64DecodingIgnoreUnknownCharacters];
  return [UIImage imageWithData:data];
}

- (UIImage *)resizeImage:(UIImage *)image tosize:(CGSize)newSize {
    UIGraphicsBeginImageContextWithOptions(newSize, NO, 0.0);
    [image drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

- (void)runModelEnquadramento:(UIImage *)image{
    dispatch_async(dispatch_get_main_queue(), ^{
        TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];
        NSDictionary *inference = (NSDictionary *)[self->model runOn:buffer];
        NSLog(@"INFERENCE: %@",inference);
        NSDictionary<NSString*,NSNumber*> *classification = inference[@"output"];
        __block NSString *highKey;
        __block NSNumber *highVal;
        [classification enumerateKeysAndObjectsUsingBlock:^(NSString *key, NSNumber *val, BOOL *stop) {
            if (highVal == nil || [val compare:highVal] == NSOrderedDescending) {
                highKey = key;
                highVal = val;
            }
        }];
        
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:highKey];
        [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
    });
}

- (void)runModelQuality:(UIImage *)image{
    dispatch_async(dispatch_get_main_queue(), ^{
        TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];
        NSDictionary *inference = (NSDictionary *)[self->model runOn:buffer];
        NSArray *result = inference[@"output"];
        NSLog(@"%@",result);
        if ([result[0] floatValue] > [result[1] floatValue]) {
            NSLog(@"QUALIDADE RUIM! Valor da inferência: %@",result[0]);
            pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"false"];
            [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
        } else {
            NSLog(@"QUALIDADE BOA! Valor da inferência: %@",result[1]);
            pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"true"];
            [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
        }
    });
}

- (void)runModelunet_vehicle_model:(UIImage *)image{
    dispatch_async(dispatch_get_main_queue(), ^{
        dispatch_async(dispatch_get_main_queue(), ^{
            TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];
            NSDictionary<TIOData> *resultDict = (NSDictionary *)[self->model runOn:buffer];
            NSArray *pixelArray = resultDict[@"output"];
            
            //upper lines from the frame
            NSArray *horizontalBorder1 = [pixelArray subarrayWithRange:NSMakeRange(0, 224)];
            NSArray *horizontalBorder2 = [pixelArray subarrayWithRange:NSMakeRange(224, 224)];
            NSArray *horizontalBorder3 = [pixelArray subarrayWithRange:NSMakeRange(448, 224)];
            
            //lines below from the frame
            NSArray *horizontalBorder4 = [pixelArray subarrayWithRange:NSMakeRange(49504, 224)];
            NSArray *horizontalBorder5 = [pixelArray subarrayWithRange:NSMakeRange(49728, 224)];
            NSArray *horizontalBorder6 = [pixelArray subarrayWithRange:NSMakeRange(49952, 224)];
            
            //Checking detection on horizontal lines...
            if ([self checkLines:horizontalBorder1]||[self checkLines:horizontalBorder2]||[self checkLines:horizontalBorder3]||[self checkLines:horizontalBorder4]||[self checkLines:horizontalBorder5]||[self checkLines:horizontalBorder6]) {
                pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"BAD_IMAGE"];
                [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
            }
            
            //Vertical lines from the frame left
            NSMutableArray *verticalBorder1 = [[NSMutableArray alloc] init];
            NSMutableArray *verticalBorder2 = [[NSMutableArray alloc] init];
            NSMutableArray *verticalBorder3 = [[NSMutableArray alloc] init];
            
            //Vertical lines from the frame right
            NSMutableArray *verticalBorder4 = [[NSMutableArray alloc] init];
            NSMutableArray *verticalBorder5 = [[NSMutableArray alloc] init];
            NSMutableArray *verticalBorder6 = [[NSMutableArray alloc] init];
            
            //Filling in the vertical arrays...
            for (int index = 0; index<224; index++) {
                NSArray *line = [pixelArray subarrayWithRange:NSMakeRange(index*224, 224)];
                    [verticalBorder1 addObject:line[0]];
                    [verticalBorder2 addObject:line[1]];
                    [verticalBorder3 addObject:line[2]];
                    [verticalBorder4 addObject:line[221]];
                    [verticalBorder5 addObject:line[222]];
                    [verticalBorder6 addObject:line[223]];
            }
            
            //Checking detection on vertical lines...
            if ([self checkLines:verticalBorder1]||[self checkLines:verticalBorder2]||[self checkLines:verticalBorder3]||[self checkLines:verticalBorder4]||[self checkLines:verticalBorder5]||[self checkLines:verticalBorder6]){
                pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"BAD_IMAGE"];
                [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
            }
            //Nothing detected, picture ok
            else {
                pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"GOOD_IMAGE"];
                [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
            }
        });
    });
}

- (BOOL)checkLines:(NSArray*)line{
    int pixelsInLine = 0;
    for (int index = 0; index < 224; index++) {
        float pixel = [line[index] floatValue];
        if (pixel >= 0.5) {
            pixelsInLine++;
        } else {
            pixelsInLine = 0;
        }
        if (pixelsInLine == 10) {
            return true;
        }
    }
    return false;
}

@end
