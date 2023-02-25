import React, {useRef, useEffect} from 'react';
import Webcam from "react-webcam";
import { Camera } from "@mediapipe/camera_utils";
import YoloDetector from "./detector"
import { setupStats } from './StatsPanel';
import { drawDetections } from './util';

export function ObjectDetection(){
    const webcamRef = useRef(null);
    const detector = useRef(null);
    const canvasRef = useRef(null);
    let stats = new setupStats();
    let startInferenceTime, numInferences = 0;
    let inferenceTimeSum = 0, lastPanelUpdate = 0;

    const beginEstimateDetectionStats =() =>{
        startInferenceTime = (performance || Date).now();
    }

    const endEstimateDetectionStats =() =>{
        const endInferenceTime = (performance || Date).now();
        inferenceTimeSum += endInferenceTime - startInferenceTime;
        ++numInferences;
    
        const panelUpdateMilliseconds = 1000;
        if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
            const averageInferenceTime = inferenceTimeSum / numInferences;
            inferenceTimeSum = 0;
            numInferences = 0;
            stats.customFpsPanel.update(
                1000.0 / averageInferenceTime, 120 /* maxValue */);
            lastPanelUpdate = endInferenceTime;
        }
    }

    const renderResult = async (video)=>{
        let detections;
        if(detector !== null){
            beginEstimateDetectionStats();
            try{
                detections = await detector.current.detect(video);
            }catch(error){
                detector.current.dispose();
                detector = null;
                alert(error);
            }
            endEstimateDetectionStats();
        }

        const canvasElement = canvasRef.current;
        const canvasCtx = canvasElement.getContext('2d')
        canvasElement.width = 640;
        canvasElement.height = 480;
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if(detections && detections.length > 0){
            drawDetections(detections,canvasCtx);
        }
        canvasCtx.restore();
    }

    useEffect(()=>{
        const loadDetector = async () =>{
          detector.current = new YoloDetector(`${process.env.PUBLIC_URL}/model/yoloxnanof32_640x480`);
          await detector.current.setModelBackend("wasm");
          await detector.current.load();
        }
        loadDetector();
      }, []);

    const runDetection =() =>{
        const camera = new Camera(webcamRef.current.video,{
            onFrame: async () =>{
                await renderResult(webcamRef.current.video);
            },
            facingMode:"user",
            width: 640,
            height: 480
        });
        camera.start();
    }

    return(
        <>
        <Webcam ref = {webcamRef} />
        <canvas
        ref={canvasRef}
        className="output_canvas"
        style={{
            position: "fixed",
            left: 640,
            top: 0,
            width: 640,
            height: 480,
        }}
        >
        </canvas>
        <button onClick={runDetection}>turn on model</button>
        </>
    )
}