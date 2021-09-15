using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MovieRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // to call LoadData() method and return the Train and Test data
            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

            EvaluateModel(mlContext, testDataView, model);

            SaveModel(mlContext, trainingDataView.Schema, model);

        }

        // IDataView is a flexible, efficient way of describing tabular data (numeric and text).
        // Data can be loaded from a text file or in real time (for example, SQL database or log files) to an IDataView object.
        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {

            // MapValueToKey() method to transform each userId and each movieId into a numeric key type Feature column
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            // ======================================================== //
            // userId	movieId	Label	userIdEncoded	movieIdEncoded  //
            // ======================================================== //
            //   1         1      4        userKey1        movieKey1    //
            //   1         3      4        userKey1        movieKey2    //
            //   1         6      4        userKey1        movieKey3    //
            // ======================================================== //

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 100,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            // ==================================================================
            //  Matrix Factorization is a common approach to recommendation
            //  when you have data on how users have rated products in the past,
            //  which is the case for the datasets in this tutorial.
            // ==================================================================

            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;

        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {

            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {

            Console.WriteLine("=============== Making a prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            var testInput = new MovieRating { userId = 6, movieId = 10 };

            var movieRatingPrediction = predictionEngine.Predict(testInput);

            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine("Movie " + testInput.movieId + " is recommended for user " + testInput.userId);
            }
            else
            {
                Console.WriteLine("Movie " + testInput.movieId + " is not recommended for user " + testInput.userId);
            }

        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {

            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);

        }

    }
}
