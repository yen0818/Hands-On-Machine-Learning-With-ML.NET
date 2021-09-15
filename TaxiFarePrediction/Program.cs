using System;
using System.IO;
using Microsoft.ML;

namespace TaxiFarePrediction
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            // to declare and initialize the mlContext variable
            MLContext mlContext = new MLContext(seed: 0);

            // call the Train method
            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext, model);

        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // Loads the data
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

            // Extracts and transforms the data
            // the FareAmount column is the Label that you will predict (the output of the model)
            // Use the CopyColumnsEstimator transformation class to copy FareAmount
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                // OneHotEncodingTransformer transformation class - categorical to numerical
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                // combines all of the feature columns into "Features" column
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                // Append the FastTreeRegressionTrainer machine learning task to the data transformation
                .Append(mlContext.Regression.Trainers.FastTree());

            // Fit the model to the training dataview and return the trained mode
            var model = pipeline.Fit(dataView);
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // Loads the test dataset          
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            
            // Creates the regression evaluator
            var predictions = model.Transform(dataView);

            // Evaluates the model and creates metrics
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            // Displays the metrics
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");

            // RSquared is another evaluation metric of the regression models.
            // The closer its value is to 1, the better the model is.
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");

            // RMS is one of the evaluation metrics of the regression model.
            // The lower it is, the better the model is.
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            // Creates a single comment of test data.

            // Predicts fare amount based on test data.
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            // Combines test data and predictions for reporting.
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            // Displays the predicted results.
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");

        }

    }
}
