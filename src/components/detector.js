import React from 'react';
import "@tensorflow/tfjs-backend-wasm"
import * as tf from "@tensorflow/tfjs"

class YoloDetector extends React.Component{
    constructor(modelPath){
        this.modelPath = modelPath;
        this.state = {
            model: null
        }
    }

};