import React, {useRef, useEffect} from 'react';
import Webcam from "react-webcam";
import { Camera } from "@mediapipe/camera_utils";

export function ObjectDetection(){
    const webcamRef = useRef(null);

    useEffect(()=>{
        const camera = new Camera(webcamRef.current.video,{
            onFrame: async () =>{

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