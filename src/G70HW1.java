import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import java.util.ArrayList;
import java.util.List;
public class G70HW1 {

    /**
     * Handles Spark configuration, data loading from a CSV file into a JavaPairRDD,
     * orchestration of the MR-Fair-FFT algorithm, and evaluation of the clustering objective.
     * @param args [filePath, kA, kB, L]
     */
    public static void main (String[] args) {
        if (args.length != 4)
            throw new IllegalArgumentException("USAGE: file_path KA KB L");

        String filePath = args[0];
        int kA = Integer.parseInt(args[1]);
        int kB = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);

        SparkConf conf = new SparkConf(true).setAppName("G70HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        System.out.println("File path = " + filePath + ", KA = " + kA + ", KB = " + kB + ", L = " + L);

        JavaPairRDD<Vector, String> inputPoints = sc.textFile(filePath).repartition(L).mapToPair(G70HW1::parseLine).cache();

        long N = inputPoints.count();
        long NA = inputPoints.filter(p->p._2().equals("A")).count();
        long NB = inputPoints.filter(p->p._2().equals("B")).count();
        System.out.println("N = " + N + ", NA = " + NA + ", NB = " + NB);

        long start = System.currentTimeMillis();
        ArrayList<Tuple2<Vector, String>> solution = MRFairFFT(inputPoints, kA, kB);
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

    /**
     * Sequential implementation of the Fair Farthest-First Traversal (Fair-FFT).
     * Selects k = kA + kB centers one by one, always picking the point farthest from those already chosen,
     * while keeping track of the demographic group budgets for groups A and B.
     * @param P The list of points (Vector) and their labels (String)
     * @param kA Budget for group A centers
     * @param kB Budget for group B centers
     * @return A list of selected centers
     */
    public static ArrayList<Tuple2<Vector, String>> FairFFT(List<Tuple2<Vector, String>> P, int kA, int kB) {
        int n = P.size();
        ArrayList<Tuple2<Vector, String>> centers = new ArrayList<>();
        double[] minDistances = new double[n];
        for (int i = 0; i < n; i++) minDistances[i] = Double.POSITIVE_INFINITY;
        int countA = 0, countB = 0;
        for (int i = 0; i < (kA + kB); i++) {
            int bestIndex = -1;
            double maxDist = -1.0;
            for (int j = 0; j < n; j++) {
                String label = P.get(j)._2();
                boolean isLegal = (label.equals("A") && countA < kA) || (label.equals("B") && countB < kB);
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

    /**
     * Distributed 2-round algorithm for Fair k-center clustering.
     * Round 1: Extracts local coresets by running FairFFT on each partition independently.
     * Round 2: Collects local centers to the driver and runs FairFFT on the combined coreset.
     * Memory used per partition is proportional to N/L.
     * @param P Distributed RDD of points and labels
     * @param kA Total budget for group A centers
     * @param kB Total budget for group B centers
     * @return Final set of kA + kB centers
     */
    public static ArrayList<Tuple2<Vector, String>> MRFairFFT(JavaPairRDD<Vector, String> P, int kA, int kB) {
        List<Tuple2<Vector, String>> coreset = P.mapPartitionsToPair(partition -> {
            List<Tuple2<Vector, String>> points = new ArrayList<>();
            while (partition.hasNext()) points.add(partition.next());
            return FairFFT(points, kA, kB).iterator();
        }).collect();
        return FairFFT(coreset, kA, kB);
    }

    /**
     * Parses a comma-separated string into a Key-Value pair for Spark.
     * Converts the first D values into a dense Vector and the last value into a label.
     * @param line A single line from the CSV input file
     * @return A Tuple2 containing a Vector (coordinates) and a String (group label)
     */
    public static Tuple2<Vector, String> parseLine(String line) {
        String[] elements = line.split(",");
        double[] coords = new double[elements.length - 1];
        for (int i = 0; i < elements.length - 1; i++) coords[i] = Double.parseDouble(elements[i]);
        return new Tuple2<>(Vectors.dense(coords), elements[elements.length - 1]);
    }

    /**
     * Calculates the clustering objective function: the maximum distance between
     * any point in the dataset and its nearest center in the solution set S.
     * @param points The full distributed dataset U
     * @param centers The final solution set S
     * @return The maximum distance (radius)
     */
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
