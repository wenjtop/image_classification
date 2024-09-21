import java.io.File;
public class java_onnx_inference {
    static {
        String filePath = "/home/wenjtop/myProject/image_classification/export/onnx_examples/java/build/libinference.so";
        System.load(filePath);
    }
    public static void main(String[] args) {
        String imgPath = "./dog.jpg";
        System.out.println(inference(imgPath));
    }
    public static native String inference(String imgPath);
}
