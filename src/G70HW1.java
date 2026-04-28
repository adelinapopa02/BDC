import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import java.util.ArrayList;
import java.util.List;
public class G70HW1 {

    public static void main (String[] args){
        if (args.length != 4)
            throw new IllegalArgumentException("USAGE: file_path KA KB L");

        String filePath = args[0];
        int KA = Integer.parseInt(args[1]);
        int KB = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);

        SparkConf conf = new SparkConf(true).setAppName("G70HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        System.out.println("File path = " + filePath + ", KA = " + KA + ", KB = " + KB + ", L = " + L);

        JavaPairRDD<Vector, String> inputPoints = sc.textFile(filePath).repartition(L).mapToPair(G70HW1::parseLine).cache();

        long N = inputPoints.count();
        long NA = inputPoints.filter(p->p._2().equals("A")).count();
        long NB = inputPoints.filter(p->p._2().equals("B")).count();
        System.out.println("N = " + N + ", NA = " + NA + ", NB = " + NB);

    }

    public static Tuple2<Vector, String> parseLine(String line) {
        String[] parts = line.split(",");
        double[] coords = new double[parts.length - 1];
        for (int i = 0; i < parts.length - 1; i++) coords[i] = Double.parseDouble(parts[i]);
        return new Tuple2<>(Vectors.dense(coords), parts[parts.length - 1]);
    }
}
