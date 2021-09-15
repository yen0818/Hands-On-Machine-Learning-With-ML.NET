using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;


namespace SentimentAnalysis
{
    class Program
    {
        // to create a field to hold the recently downloaded dataset file path
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt"); 
        
        static void Main(string[] args)
        {
            // to declare and initialize the mlContext variable
            MLContext mlContext = new MLContext();

            TrainTestData splitDataView = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            Evaluate(mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(mlContext, model);

            UseModelWithBatchItems(mlContext, model);

        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            // Loads the data.
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            // The LoadFromTextFile() method defines the data schema and reads in the file.
            // It takes in the data path variables and returns an IDataView.

            // Splits the loaded dataset into train and test datasets.
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // Returns the split train and test datasets.
            return splitDataView;
        }
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // Extracts and transforms the data.
            // 1st Append - appended to the estimator and accepts the featurized SentimentText (Features) and the Label input parameters to learn from the historic data
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            // Trains the model.
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            // Predicts sentiment based on test data.

            // Returns the model.
            return model;
        }
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            // Loads the test dataset.
            // Creates the BinaryClassification evaluator.
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);

            // Evaluates the model and creates metrics.
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            // Displays the metrics.
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            
            // Creates a single comment of test data.
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };

            // Predicts sentiment based on test data.
            // Combines test data and predictions for reporting.
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            // Displays the predicted results.
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();

        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            // Creates batch test data.
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };

            // Predicts sentiment based on test data.
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            // Combines test data and predictions for reporting.
            // Displays the predicted results.
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }

    }
}
