import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import java.util.ArrayList;
import java.util.List;
public class G70HW1 {

    public static void main (String[] args) {
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

        long start = System.currentTimeMillis();
        ArrayList<Tuple2<Vector, String>> solution = MRFairFFT(inputPoints, KA, KB, L);
        long end = System.currentTimeMillis();

        for (Tuple2<Vector, String> center : solution) {
            double[] coords = center._1().toArray();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < coords.length; i++) {
                sb.append(coords[i]).append(i == coords.length - 1 ? "" : ",");
            }
            System.out.println("Center = [" + sb + "] Label = " + center._2());
        }

        double objective = computeObjective(inputPoints, solution);
        System.out.println("Objective function = " + objective);
        System.out.println("Running time of MRFairFFT = " + (end - start) + " ms");
    }

    public static Tuple2<Vector, String> parseLine(String line) {
        String[] elements = line.split(",");
        double[] coords = new double[elements.length - 1];
        for (int i = 0; i < elements.length - 1; i++) coords[i] = Double.parseDouble(elements[i]);
        return new Tuple2<>(Vectors.dense(coords), elements[elements.length - 1]);
    }

    public static ArrayList<Tuple2<Vector, String>> FairFFT(List<Tuple2<Vector, String>> P, int KA, int KB) {
        int n = P.size();
        ArrayList<Tuple2<Vector, String>> centers = new ArrayList<>();
        double[] minDistances = new double[n];
        for (int i = 0; i < n; i++) minDistances[i] = Double.POSITIVE_INFINITY;
        int countA = 0, countB = 0;
        for (int i = 0; i < (KA + KB); i++) {
            int bestIndex = -1;
            double maxDist = -1.0;
            for (int j = 0; j < n; j++) {
                String label = P.get(j)._2();
                boolean isLegal = (label.equals("A") && countA < KA) || (label.equals("B") && countB < KB);
                if (isLegal && (centers.isEmpty() || minDistances[j] > maxDist)) {
                    maxDist = minDistances[j];
                    bestIndex = j;
                }
            }
            Tuple2<Vector, String> newCenter = P.get(bestIndex);
            centers.add(newCenter);
            if (newCenter._2().equals("A")) countA++; else countB ++;
            for(int j = 0; j < n; j++) {
                minDistances[j]= Math.min(minDistances[j], Vectors.sqdist(P.get(j)._1(), newCenter._1()));
            }
        }
        return centers;
    }

    public static ArrayList<Tuple2<Vector, String>> MRFairFFT(JavaPairRDD<Vector, String> P, int KA, int KB, int L) {
        List<Tuple2<Vector, String>> coreset = P.mapPartitionsToPair(partition -> {
            List<Tuple2<Vector, String>> points = new ArrayList<>();
            while (partition.hasNext()) points.add(partition.next());
            return FairFFT(points, KA, KB).iterator();
        }).collect();
        return FairFFT(coreset, KA, KB);
    }

    public static double computeObjective(JavaPairRDD<Vector, String> points, List<Tuple2<Vector, String>> centers) {
        return Math.sqrt(points.map(p -> {
            double minDistSq = Double.POSITIVE_INFINITY;
            for (Tuple2<Vector, String> c : centers) {
                minDistSq = Math.min(minDistSq, Vectors.sqdist(p._1(), c._1()));
            }
            return minDistSq;
        }).reduce(Math::max));
    }
}
