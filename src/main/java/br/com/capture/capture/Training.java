package br.com.capture.capture;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.javacpp.opencv_face.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.opencv.core.CvType.CV_32SC1;

public class Training {

    public static void main(String args[]) {

        File directory = new File("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\fotos");

        FilenameFilter filterImage = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") ||
                        name.endsWith(".gif") ||
                        name.endsWith(".png");
            }
        };

        //File [] arquivos = directory.listFiles(filterImage);

        List<File> archive = new ArrayList<>(Arrays.asList(directory.listFiles(filterImage)));
        opencv_core.MatVector photos = new opencv_core.MatVector(archive.size());

        //ROTULO DA PESSOA {ID}
        opencv_core.Mat rotulos = new opencv_core.Mat(archive.size(), 1, CV_32SC1);

        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;

        for (File image : archive) {
            opencv_core.Mat foto = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(image.getName().split("\\.")[1]);

            resize(foto, foto, new opencv_core.Size(160, 160));
            photos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }

        opencv_face.FaceRecognizer eigenfaces = createEigenFaceRecognizer();
        opencv_face.FaceRecognizer fisherfaces = createFisherFaceRecognizer();
        opencv_face.FaceRecognizer lbph = createLBPHFaceRecognizer();

        eigenfaces.train(photos,rotulos);
        eigenfaces.save("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\classificadores\\classificadorEigenFaces.yml");

        fisherfaces.train(photos,rotulos);
        fisherfaces.save("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\classificadores\\classificadorFisherFaces.yml");

        lbph.train(photos, rotulos);
        lbph.save("C:\\Projetos Pessoais\\capture_capture\\src\\main\\java\\br\\com\\capture\\recursos\\classificadores\\classificadorLBPH.yml");
    }
}
