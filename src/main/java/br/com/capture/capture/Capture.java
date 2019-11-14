package br.com.capture.capture;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.*;
import org.bytedeco.javacv.Frame;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.util.Scanner;

import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/***
 * @author Renan
 */
public class Capture {

    public static void main(String args[]) throws FrameGrabber.Exception, InterruptedException {

        KeyEvent capture = null;
        OpenCVFrameConverter.ToMat convertMat =  new OpenCVFrameConverter.ToMat();

        opencv_objdetect.CascadeClassifier detectFaces = new opencv_objdetect.CascadeClassifier("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\haarcascade-frontalface-alt.xml");

        //0 Significa minha webcam, se tivesse outra camera o numero troca
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();


        CanvasFrame cframe = new CanvasFrame("Preview",
                CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapture = null;

        opencv_core.Mat imageColor;

        int numeroAmostras = 25;
        int amostra = 1;

        int idPessoa = Integer.parseInt(JOptionPane.showInputDialog(null, "Digite seu ID","Informe o seu id", JOptionPane.WARNING_MESSAGE));
        while((frameCapture = camera.grab()) != null ) {

            imageColor = convertMat.convert(frameCapture);
            opencv_core.Mat colorGray = new opencv_core.Mat();

            cvtColor(imageColor, colorGray, COLOR_BGRA2GRAY);

            opencv_core.RectVector facesDetecteds = new opencv_core.RectVector();

            detectFaces.detectMultiScale(
                    colorGray,
                    facesDetecteds,
                    1.1,
                    1,
                    0,
                    new opencv_core.Size(150,150),
                    new opencv_core.Size(500,500)
            );

            if(capture == null) {
                capture = cframe.waitKey(5);
            }



            for (int i = 0; i < facesDetecteds.size(); i++) {
                opencv_core.Rect dadosFace = facesDetecteds.get(0);

                //DESENHANDO O RETANGULHO VERMELHO
                rectangle(
                        imageColor,
                        dadosFace,
                        new opencv_core.Scalar(0,0,255, 0)
                );

                opencv_core.Mat faceCapturada = new opencv_core.Mat(colorGray,dadosFace);
                resize(faceCapturada, faceCapturada, new opencv_core.Size(160,160));
                if(capture == null) {
                    capture = cframe.waitKey(5);
                }


                // PEGANDO A TECLA 'q' PARA TIRAR AS FOTOS E COLOCANDO NO PACOTE FOTOS COM {ID}{AMOSTA}.JPG
                if (capture != null){
                    if(capture.getKeyChar() == 'q') {
                        if(amostra <= numeroAmostras) {
                            imwrite("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\fotos\\pessoa."
                                    + idPessoa + "." + amostra + ".jpg", faceCapturada);
                            System.out.println("Foto "+ amostra + " capturada\n");
                            amostra++;
                        }
                    }
                    capture = null;
                }
            }

            if(capture == null) {
                capture = cframe.waitKey(20);
            }
            if(cframe.isVisible()) {
                cframe.showImage(frameCapture);
            }

            if(amostra > numeroAmostras) {
                break;
            }
        }
        cframe.dispose();
        camera.stop();

    }
}
