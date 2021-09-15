using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

public class IrisData
{
    // Use the LoadColumn attribute to specify the indices of the source columns in the data set file.
    [LoadColumn(0)]
    public float SepalLength;

    [LoadColumn(1)]
    public float SepalWidth;

    [LoadColumn(2)]
    public float PetalLength;

    [LoadColumn(3)]
    public float PetalWidth;
}

public class ClusterPrediction
{
    // Use the ColumnName attribute to bind the PredictedClusterId and Distances fields
    // to the PredictedLabel and Score columns respectively
    [ColumnName("PredictedLabel")] // ID of the predicted cluster
    public uint PredictedClusterId;

    [ColumnName("Score")] // An array with squared Euclidean distances to the cluster centroids.
                          // The array length is equal to the number of clusters.
    public float[] Distances;
}
