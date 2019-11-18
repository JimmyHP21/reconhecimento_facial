package br.com.capture.reconhecimento;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.*;

import javax.print.attribute.standard.Sides;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.javacpp.opencv_face.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class Reconhecimento {
    public static void main(String args[]) throws FrameGrabber.Exception {
        OpenCVFrameConverter.ToMat converterMat = new OpenCVFrameConverter.ToMat();

        opencv_objdetect.CascadeClassifier detectFaces = new opencv_objdetect.CascadeClassifier("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\haarcascade-frontalface-alt.xml");

        List<String> pessoas = new ArrayList<>(Arrays.asList("", "Renan", "Leonardo"));

        //EINGENFACES
        //opencv_face.FaceRecognizer reconhecedor = createEigenFaceRecognizer();
        //reconhecedor.load("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\classificadores\\classificadorEigenFaces.yml");
        //reconhecedor.setThreshold(3300);

        //FISHERFACES
        //opencv_face.FaceRecognizer reconhecedor = createFisherFaceRecognizer();
        //reconhecedor.load("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\classificadores\\classificadorFisherFaces.yml");

        //LBPH
        opencv_face.FaceRecognizer reconhecedor = createLBPHFaceRecognizer();
        reconhecedor.load("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\classificadores\\classificadorLBPH.yml");


        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();

        CanvasFrame cFrame = new CanvasFrame("Reconhecimento",CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapture = null;

        opencv_core.Mat imageColor = new opencv_core.Mat();

        while ((frameCapture = camera.grab()) != null) {
            imageColor = converterMat.convert(frameCapture);

            opencv_core.Mat imageGray = new opencv_core.Mat();

            cvtColor(imageColor,imageGray,COLOR_BGRA2GRAY);

            opencv_core.RectVector facesDetecteds = new opencv_core.RectVector();

            detectFaces.detectMultiScale(
                    imageGray,
                    facesDetecteds,
                    1.1,
                    2,
                    0,
                    new opencv_core.Size(150,150),
                    new opencv_core.Size(500,500)
            );

            for(int i = 0; i < facesDetecteds.size(); i++){
                opencv_core.Rect dadosFace = facesDetecteds.get(i);
                rectangle(imageColor,dadosFace, new opencv_core.Scalar(0,255,0,0));
                opencv_core.Mat faceCapturada = new opencv_core.Mat(imageGray,dadosFace);
                resize(faceCapturada, faceCapturada, new opencv_core.Size(160,160));

                IntPointer rotulo = new IntPointer(1);
                DoublePointer confianca = new DoublePointer(1);

                reconhecedor.predict(faceCapturada,rotulo,confianca);

                int predict = rotulo.get(0);
                String nome;
                if (predict == -1) {
                    nome = "Desconhecido";
                } else {
                    nome = pessoas.get(predict) + " - " + confianca.get(0);
                }

                int x = Math.max(dadosFace.tl().x() - 10, 0);
                int y = Math.max(dadosFace.tl().y() - 10, 0);

                putText(
                        imageColor,
                        nome,
                        new opencv_core.Point(x,y),
                        CV_FONT_HERSHEY_PLAIN,
                        1.4,
                        new opencv_core.Scalar(0,255,0,0)
                );

            }

            if(cFrame.isVisible()) {
                cFrame.showImage(frameCapture);
            }
        }
        cFrame.dispose();
        camera.stop();
    }
}
