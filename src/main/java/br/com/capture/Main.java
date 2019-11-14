package br.com.capture;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.*;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.awt.event.KeyEvent;

public class Main {

    public static void main(String args[]) throws FrameGrabber.Exception{

        KeyEvent capture = null;
        OpenCVFrameConverter.ToMat convertMat =  new OpenCVFrameConverter.ToMat();


        //0 Significa minha webcam, se tivesse outra camera o numero troca
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();


        CanvasFrame cframe = new CanvasFrame("Preview",
                CanvasFrame.getDefaultGamma() / camera.getGamma());

        Frame frameCapture = null;


        while((frameCapture = camera.grab()) != null ) {

            if(cframe.isVisible()) {
                cframe.showImage(frameCapture);
            }
        }
        cframe.dispose();
        camera.stop();

    }
}
