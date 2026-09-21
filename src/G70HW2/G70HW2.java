import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import java.util.*;
import java.util.concurrent.Semaphore;

public class G70HW2 {

    public static void main(String[] args) throws Exception {
        if (args.length != 7) {
            throw new IllegalArgumentException("USAGE: n phi epsilon delta d w portExp");
        }

        SparkConf conf = new SparkConf(true).setMaster("local[*]").setAppName("G70HW2");
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        // Input parameters
        int n = Integer.parseInt(args[0]);
        double phi = Double.parseDouble(args[1]);
        double epsilon = Double.parseDouble(args[2]);
        double delta = Double.parseDouble(args[3]);
        int d = Integer.parseInt(args[4]);
        int w = Integer.parseInt(args[5]);
        int portExp = Integer.parseInt(args[6]);

        System.out.println("INPUT PARAMETERS");
        System.out.println("n = " + n);
        System.out.println("phi = " + phi);
        System.out.println("epsilon = " + epsilon);
        System.out.println("delta = " + delta);
        System.out.println("d = " + d);
        System.out.println("w = " + w);
        System.out.println("port = " + portExp);

        // Stream length and exact item frequencies
        long[] streamLength = new long[1];
        streamLength[0] = 0L;
        Map<Integer, Long> trueFrequencies = new HashMap<>();

        // Sticky Sampling data structures
        double r = Math.log(1.0 / (delta * phi)) / epsilon;
        Map<Integer, Long> stickySample = new HashMap<>();
        Random rnd = new Random();

        // Count-Min Sketch data structures
        int p = 8191;
        long[][] cmSketch = new long[d][w];
        long[] aCM = new long[d], bCM = new long[d];
        for (int i = 0; i < d; i++) {
            aCM[i] = rnd.nextInt(p - 1) + 1;
            bCM[i] = rnd.nextInt(p);
        }
        Set<Integer> FCM = new HashSet<>();

        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire();

        // Process stream sequentially by collecting the raw items from each batch
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK).foreachRDD((batch, time) -> {
            if (streamLength[0] >= n) return;

            // Collect items as a sequential list
            List<String> rawBatchItems = batch.filter(s -> !s.trim().isEmpty()).collect();

            if (rawBatchItems.isEmpty()) return;

            for (String s : rawBatchItems) {
                if (streamLength[0] >= n) {
                    break;
                }

                int item = Integer.parseInt(s.trim());
                streamLength[0]++;

                trueFrequencies.merge(item, 1L, Long::sum);

                // Sticky Sampling step
                if (stickySample.containsKey(item)) {
                    stickySample.put(item, stickySample.get(item) + 1L);
                } else {
                    if (rnd.nextDouble() <= r / n) {
                        stickySample.put(item, 1L);
                    }
                }

                // Count-Min Sketch step
                long currentMinEstimate = Long.MAX_VALUE;
                for (int i = 0; i < d; i++) {
                    int col = (int) (Math.floorMod(aCM[i] * item + bCM[i], p) % w);
                    cmSketch[i][col]++;

                    if (cmSketch[i][col] < currentMinEstimate) {
                        currentMinEstimate = cmSketch[i][col];
                    }
                }

                if (!FCM.contains(item) && currentMinEstimate >= (phi * n)) {
                    FCM.add(item);
                }
            }

            if (streamLength[0] >= n) {
                stoppingSemaphore.release();
            }
        });

        sc.start();
        stoppingSemaphore.acquire();
        sc.stop(false, false);

        double threshold = phi * n;

        // Print true frequent items
        List<Integer> trueFrequent = new ArrayList<>();
        for (Map.Entry<Integer, Long> e : trueFrequencies.entrySet()) {
            if (e.getValue() >= threshold) trueFrequent.add(e.getKey());
        }
        Collections.sort(trueFrequent);

        System.out.println("\nTRUE FREQUENT ITEMS");
        for (int item : trueFrequent) {
            System.out.println("Item = " + item + " True Freq = " + trueFrequencies.get(item));
        }

        // Print F_SS
        List<Integer> FSS = new ArrayList<>();
        for (Map.Entry<Integer, Long> e : stickySample.entrySet()) {
            if (e.getValue() >= (phi - epsilon) * n) FSS.add(e.getKey());
        }
        Collections.sort(FSS);

        System.out.println("\nSTICKY SAMPLING");
        System.out.println("Size of dictionary = " + stickySample.size());
        for (int item : FSS) {
            System.out.println("Item = " + item + " True Freq = " + trueFrequencies.getOrDefault(item, 0L));
        }

        // Print F_CM
        List<Integer> fcmList = new ArrayList<>(FCM);
        Collections.sort(fcmList);

        System.out.println("\nCOUNT-MIN SKETCH");
        System.out.println("Size of F_CM = " + fcmList.size());
        for (int item : fcmList) {
            System.out.println("Item = " + item + " True Freq = " + trueFrequencies.getOrDefault(item, 0L));
        }
    }
}