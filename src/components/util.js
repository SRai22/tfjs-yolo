import React from 'react'
import * as tf from "@tensorflow/tfjs"

export const COCOLabels = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
  ];
  

export class DetectedObject extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            classId: 0,
            bbox: [0, 0, 0, 0], // x, y, width, height
            className: '',
            score: 0.0
        };
    }
    render(){
        return(
            <>
            <p>BoundingBox: {this.state.bbox} </p>
            <p>ClassName: {this.state.className}</p>
            <p>Score: {this.state.score} </p>
            </>
        );
    }

}

function CalculateIoU(obj0, obj1) {
    const interx0 = Math.max(obj0[0], obj1[0]);
    const intery0 = Math.max(obj0[1], obj1[1]);
    const interx1 = Math.min(obj0[0] + obj0[2], obj1[0] + obj1[2]);
    const intery1 = Math.min(obj0[1] + obj0[3], obj1[1] + obj1[3]);
    if (interx1 < interx0 || intery1 < intery0) {
        return 0;
    }

    const area0 = obj0[2] * obj0[3];
    const area1 = obj1[2] * obj1[3];
    const areaInter = (interx1 - interx0) * (intery1 - intery0);
    const areaSum = area0 + area1 - areaInter;

    return areaInter / areaSum;
}

function compareScores(lhs, rhs){
    if(lhs.state.score < rhs.state.score){
        return 1;
    }
    if(lhs.state.score > rhs.state.score){
        return -1;
    }
    return 0;
}

export function NMS(detectionList, nmsIouThreshold, checkClassId) {
    const nmsDectectionList = []
    detectionList.sort((a,b)=> b.state.score - a.state.score);
    const isMerged = new Array(detectionList.length).fill(false);
    for (let indexHighScore = 0; indexHighScore < detectionList.length; indexHighScore++) {
        let candidates = [];
        if (isMerged[indexHighScore]) continue;
        candidates.push(detectionList[indexHighScore]);
        for (let indexLowScore = indexHighScore + 1; indexLowScore < detectionList.length; indexLowScore++) {
            if (isMerged[indexLowScore]) continue;
            if (checkClassId && detectionList[indexHighScore].state.classId !== 
                                detectionList[indexLowScore].state.classId) continue;
            if (CalculateIoU(detectionList[indexHighScore].state.bbox, 
                             detectionList[indexLowScore].state.bbox) > nmsIouThreshold) {
                //candidates.push(detectionList[indexLowScore]);
                let det = new DetectedObject(); 
                det.state.bbox[0] = Math.min(detectionList[indexHighScore].state.bbox[0], detectionList[indexLowScore].state.bbox[0]);
                det.state.bbox[1] = Math.min(detectionList[indexHighScore].state.bbox[1], detectionList[indexLowScore].state.bbox[1]);
                det.state.bbox[2] = Math.max(detectionList[indexHighScore].state.bbox[2], detectionList[indexLowScore].state.bbox[2]);
                det.state.bbox[3] = Math.max(detectionList[indexHighScore].state.bbox[3], detectionList[indexLowScore].state.bbox[3]);
                det.state.score = Math.max(detectionList[indexHighScore].state.score, detectionList[indexLowScore].state.score);
                det.state.classId = detectionList[indexHighScore].state.classId;
                det.state.className = detectionList[indexHighScore].state.className;
                candidates.push(det);
                isMerged[indexLowScore] = true;
            }
        }

        nmsDectectionList.push(candidates[0]);
    }

    return nmsDectectionList;
}

/**
 * Color array
 */
const COLOR_PALETTE = [
    '#ffffff', '#800000', '#469990', '#e6194b', '#42d4f4', '#fabed4', '#aaffc3',
    '#9a6324', '#000075', '#f58231', '#4363d8', '#ffd8b1', '#dcbeff', '#808000',
    '#ffe119', '#911eb4', '#bfef45', '#f032e6', '#3cb44b', '#a9a9a9'
  ];
  
export const drawDetections = (detections, ctx) =>{
    // Loop through each bounding box
    let i=0;
    detections.forEach(prediction => {
      // Extract boxes and classes
      const [x, y, width, height] = prediction.state.bbox; 
      const text = prediction.state.className; 
      if(text){
        // Set styling
        const color = COLOR_PALETTE[i]
        ctx.strokeStyle = color
        ctx.font = '18px Arial';
    
        // Draw rectangles and text
        ctx.beginPath();   
        ctx.fillStyle = color
        ctx.fillText(text, x, y);
        ctx.rect(x, y, width, height); 
        ctx.stroke();
        i++;
      }
    });
}