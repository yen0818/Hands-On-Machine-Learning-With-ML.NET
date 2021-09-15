using System;
using System.IO;
using Microsoft.ML;
using System.Collections.Generic;

namespace ProductSalesAnomalyDetection
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
        //assign the Number of records in dataset file to constant variable
        const int _docsize = 36; // _docsize to calculate pvalueHistoryLength
        
        // The CreateEmptyDataView() produces an empty data view object with the correct schema
        // to be used as input to the IEstimator.Fit() method.
        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            // Create empty DataView. We just need the schema to call Fit() for the time series transforms
            IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // load the data
            // The LoadFromTextFile() defines the data schema and reads in the file. 
            // It takes in the data path variables and returns an IDataView.
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            // ====================================================================================
            // Anomaly detection is the process of detecting time-series data outliers
            // There are two types of time series anomalies that can be detected:
            // Spikes - indicate temporary bursts of anomalous behavior in the system.
            // Change points - indicate the beginning of persistent changes over time in the system.
            // ====================================================================================

            DetectSpike(mlContext, _docsize, dataView);

            DetectChangepoint(mlContext, _docsize, dataView);
        }

        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            // Use the IidSpikeEstimator to train the model for spike detection.
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, pvalueHistoryLength: docSize / 4);

            // Create the spike detection transform
            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));

            //  To transform the productSales data
            //  Transform() method to make predictions for multiple input rows of a dataset
            IDataView transformedData = iidSpikeTransform.Transform(productSales);

            // Convert your transformedData into a strongly typed IEnumerable for easier display
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value");

            // ====================================================================================
            // Alert - indicates a spike alert for a given data point.
            // Score - is the ProductSales value for a given data point in the dataset.
            // P - Value - The "P" stands for probability.The closer the p - value is to 0, the more likely the data point is an anomaly.
            // ====================================================================================

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }

                Console.WriteLine(results);
            }
            Console.WriteLine("");

        }

        static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, changeHistoryLength: docSize / 4);

            var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));

            IDataView transformedData = iidChangePointTransform.Transform(productSales);

            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

            // ====================================================================================
            // Martingale value - is used to identify how "weird" a data point is,
            // based on the sequence of P-values.
            // ====================================================================================

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- alert is on, predicted changepoint";
                }
                Console.WriteLine(results);
            }
            Console.WriteLine("");

        }

    }
}
