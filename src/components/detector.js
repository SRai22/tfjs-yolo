import React from 'react';
import * as tfjsWasm from "@tensorflow/tfjs-backend-wasm"
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {COCOLabels, DetectedObject, NMS} from "./util";

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

const GridScaleList = [8, 16, 32]
const GridChannel = 1;
const NumberOfClass = 80;
const ElementNumOfAnchor = NumberOfClass + 5; 

class YoloDetector extends React.Component{
    constructor(modelPath){
        super(modelPath);
        this.modelPath = modelPath;
        this.model = null;
        this.modelWidth = 640;
        this.modelHeight = 480;
        this.boxConfidenceThreshold = 0.5;
        this.classConfidenceThreshold = 0.4;
        this.nmsIouThreshold = 0.5;
        this.detectionsList = [];
    }

    async setModelBackend(backend){
        await tf.setBackend(backend);
        await tf.ready();
    }

    async load() {
        this.model = await tfconv.loadGraphModel(`${this.modelPath}/model.json`);
        while(this.model === null);
        const zeroTensor = tf.zeros([1, this.modelHeight, this.modelWidth, 3], 'float32');
        // Warmup the model.
        const result = this.model.execute(zeroTensor);
        if (result.shape.length !== 3 || result.shape[0] !== 1) {
            result.dispose();
            throw new Error(
              `Unexpected output shape from model: [${result.shape}]`);
        }
        result.dispose();
        zeroTensor.dispose();
        console.log("Model ready!");
    }

    normalizeTensor(tensor){
        tensor = tf.cast(tensor, 'float32');
        //Define mean and norm values
        const meanValue = tf.tensor1d([0.485, 0.456, 0.406],"float32");
        const normValue = tf.tensor1d([0.229, 0.224, 0.225],"float32");
        return tf.div(tf.sub(tensor, meanValue), normValue);
        //return tf.cast(tensor, 'float32');
    }

    getBoundingBox(data, idx ,scale_x, scale_y, gridWidth, gridHeight){
        let index = idx;
        for(var grid_y= 0; grid_y < gridHeight; grid_y++){
            for(var grid_x=0; grid_x < gridWidth; grid_x++){
                for(var grid_c=0; grid_c < GridChannel; grid_c++){
                    const box_confidence = data[index+4];
                    if(box_confidence > this.boxConfidenceThreshold){
                        const classProbabilities = data.slice(index + 5, index + NumberOfClass + 5);
                        const confidence = Math.max(...classProbabilities);
                        const classId = classProbabilities.indexOf(confidence);
                        if((confidence*box_confidence) >= this.classConfidenceThreshold){
                            const cx = Math.floor((data[index+0]+grid_x) * scale_x);
                            const cy = Math.floor((data[index+1]+grid_y) * scale_y);
                            const w =  Math.floor((Math.exp(data[index+2]))* scale_x);
                            const h =  Math.floor((Math.exp(data[index+3]))* scale_y);
                            const x =  cx - w /2 ;
                            const y =  cy - h /2 ;
                            console.log("cx", cx, "cy", cy, "w", w, "h", h);
                            let detection = new DetectedObject();
                            detection.state.classId = classId;
                            detection.state.bbox = [x, y, w, h];
                            detection.state.className = COCOLabels[classId];
                            detection.state.score = confidence;
                            this.detectionsList.push(detection);
                        }
                    }
                    index = index + ElementNumOfAnchor;
                } 
            }
        }
    }

    /**
    * Infers through the model.
    *
    * @param img The image to classify. Can be a tensor or a DOM element image,
    * video, or canvas.
    * @param maxNumBoxes The maximum number of bounding boxes of detected
    * objects. There can be multiple objects of the same class, but at different
    * locations. Defaults to 20.
    * @param minScore The minimum score of the returned bounding boxes
    * of detected objects. Value between 0 and 1. Defaults to 0.5.
    * 
    * Normalize the tensors before feeding to the network
    */
    async infer(img){
        const batched = tf.tidy(() => {
            if (!(img instanceof tf.Tensor)) {
              img = tf.browser.fromPixels(img);
              img = this.normalizeTensor(img);
            }
            // Reshape to a batch
            return tf.expandDims(this.normalizeTensor(img));
        });
        console.log("batch-shape",batched.shape);
        const height = batched.shape[1];
        const width = batched.shape[2];
        // model returns one tensor:
        // The shape of the output tensor is [1,6300,85]
        // where 6300 is the number of detections, 80 is the number of classes.
        // and 4 is the four coordinates of the box.
        const outputTensor = this.model.execute(batched);
        // Only use asynchronous downloads when we really have to (WebGPU) because
        // that will poll for download completion using setTimeOut which introduces
        // extra latency.
        let inferenceResult;
        if (tf.getBackend() !== 'webgpu') {
            inferenceResult = outputTensor.dataSync();
        } else {
            inferenceResult = await outputTensor.data();
        }
        inferenceResult = new Float32Array(inferenceResult.buffer);
        this.detectionsList = [];
        let idx = 0
        GridScaleList.forEach(gridScale=>{
            const gridWidth  = width/ gridScale;
            const gridHeight = height/ gridScale;
            const scale_x = gridScale ;
            const scale_y = gridScale ;
            this.getBoundingBox(inferenceResult, idx, scale_x, scale_y, gridWidth, gridHeight);
            idx = idx + (gridWidth*gridHeight*GridChannel*ElementNumOfAnchor);
        })
        console.log("det",this.detectionsList.length)
        const nmsDetectionList = NMS(this.detectionsList, this.nmsIouThreshold, false);
        console.log("nms",nmsDetectionList.length);
        return nmsDetectionList;

    }
   /**
   * Detect objects for an image returning a list of bounding boxes with
   * assocated class and score.
   *
   * @param img The image to detect objects from. Can be a tensor or a DOM
   *     element image, video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   * @param minScore The minimum score of the returned bounding boxes
   * of detected objects. Value between 0 and 1. Defaults to 0.5.
   */
   async detect(image, maxNumBoxes, minScore){
    if (image === null) {
        this.reset();
        return [];
    }

    return this.infer(image);
   }
   
   /**
   * Dispose the tensors allocated by the model. You should call this when you
   * are done with the model.
   */
    dispose() {
    if (this.model != null) {
      this.model.dispose();
    }
  }

};

export default YoloDetector;