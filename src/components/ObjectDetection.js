import React, {useRef, useEffect} from 'react';
import Webcam from "react-webcam";
import { Camera } from "@mediapipe/camera_utils";
import YoloDetector from "./detector"

export function ObjectDetection(){
    const webcamRef = useRef(null);
    const detector = useRef(null);
    let detections;

    useEffect(()=>{
        const loadDetector = async () =>{
          detector.current = new YoloDetector(`${process.env.PUBLIC_URL}/model/yoloxnanof32_640x480`);
          await detector.current.setModelBackend("wasm");
          await detector.current.load();
        }
        loadDetector();
      }, []);

    useEffect(()=>{
        const camera = new Camera(webcamRef.current.video,{
            onFrame: async () =>{
                detections = await detector.current.detect(webcamRef.current.video);
            },
            facingMode:"user",
            width: 640,
            height: 480
        });
        camera.start();
    })

    return(
        <>
        <Webcam ref = {webcamRef} />
        </>
    )
}