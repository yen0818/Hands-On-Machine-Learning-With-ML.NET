using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

public class ProductSalesData
{
    // The LoadColumn attribute specifies which columns (by column index)
    // in the dataset should be loaded.
    [LoadColumn(0)]
    public string Month;

    [LoadColumn(1)]
    public float numSales;
}

public class ProductSalesPrediction
{
    //vector to hold alert,score,p-value values
    [VectorType(3)]
    public double[] Prediction { get; set; }
    // For anomaly detection, the prediction consists of an alert to indicate
    // whether there is an anomaly, a raw score, and p-value.
    // The closer the p-value is to 0, the more likely an anomaly has occurred.
}
