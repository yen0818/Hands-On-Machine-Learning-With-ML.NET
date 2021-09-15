using System;
using System.IO; // To make the preceding code compile
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IrisFlowerClustering
{
    class Program
    {
        // _dataPath contains the path to the file with the data set used to train the model.
        // _modelPath contains the path to the file where the trained model is stored.
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            // load the data
            // Explanation: The generic MLContext.Data.LoadFromTextFile extension method infers
            // the data set schema from the provided IrisData type and
            // returns IDataView which can be used as input for transformers.
            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

            // concatenate loaded columns into one Features column
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                // use a KMeansTrainer trainer to train the model using the k-means++ clustering algorithm
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3)); // split in three clusters

            // train the model
            var model = pipeline.Fit(dataView);

            //save the model for future .NET usage
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            // Explanation for making oredections : use the PredictionEngine<TSrc,TDst> class that
            // takes instances of the input type through the transformer pipeline and produces instances of the output type
            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);
            // The PredictionEngine is a convenience API, which allows you to perform a prediction on a single instance of data.
            // PredictionEngine is not thread-safe. It's acceptable to use in single-threaded or prototype environments.
            // For improved performance and thread safety in production environments, use the PredictionEnginePool service,
            // which creates an ObjectPool of PredictionEngine objects for use throughout your application

            var prediction = predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");

        }
    }
}
