/**
 * Authors: Kazuya Gokita (@kazoo04)
 */

import std.stdio;
import std.math;
import std.file;
import std.stream;
import std.string;
import std.conv;
import std.random;


struct feature {
  uint index;
  double weight;
}

class example {
  int label;    //(-1, +1)
  feature[] fv;
}

/***
 * Adaptive Regularization of Weight Vectors
 *
 * See_Also:
 *   K. Crammer, A. Kulesza, and M. Dredze. "Adaptive regularization of weight vectors" NIPS 2009
 */
class Arow {
  private:
    size_t size;      /// Dimention
    double[] mean;    /// Average vector: μ
    double[] cov;     /// Variance Matrix: ∑ (diagonal matrix)
    immutable double hyperparameter = 0.1; ///Hyper parameter: r (r > 0)

    invariant()
    {
      assert(size > 0);
      assert(mean != null);
      assert(cov != null);
      assert(mean.length == size);
      assert(cov.length == size);
      assert(hyperparameter > 0);
    }

  public:
    this(size_t num_feature) {
      size = num_feature;
      mean = new double[size];
      cov = new double[size];

      for(int i = 0; i < size; i++) {
        mean[i] = 0.0;
        cov[i] = 1.0;
      }
    }


    /**
     * Calculate the distance between a vector and the hyperplane
     * Params:
     *  fv =  feature
     * Returns: Margin(Euclidean distance)
     */
    double GetMargin(feature[] fv)
      in
      {
        assert(fv != null);
      }
      out (result)
      {
        assert(result != double.nan);
      }
      body
      {
        // margin = x_t^T μ_t
        double margin = 0.0;

        // inner product
        foreach(v; fv) {
          margin += mean[v.index] * v.weight;
        }

        return margin;
      }


    /**
     * Calculate confidence
     * Params:
     *  fv =  feature
     *
     * Returns: confidence
     */ 
    double GetConfidence(feature[] fv)
      // confidence = x_t^T ∑_{t-1} x_t
      in 
      {
        assert(fv != null);
      }
      out(result)
      {
        assert(result != double.nan);
      }
      body
      {
        //calculate confidence
        double confidence = 0.0;
        foreach(v; fv) {
          confidence += cov[v.index] * v.weight * v.weight;
        }

        return confidence;
      }


    /**
     * Update weight vector
     * Params:
     *  fv    = feature
     *  label = class label (+1, -1)
     * Returns: loss (0 | 1)
     */
    int Update(feature[] fv, int label) 
      in
      {
        assert(label == -1 || label == +1);
        assert(fv != null);
      }
      out (result)
      {
        assert(result == 0 || result == 1);
      }
      body
      {
        double m = GetMargin(fv);

        if(m * label >= 1) return 0;

        double confidence = GetConfidence(fv);
        double beta = 1.0 / (confidence + hyperparameter);
        double alpha = (1.0 - label * m) * beta;

        //Update mean(μ)
        foreach (v; fv) {
          mean[v.index] += alpha * cov[v.index] * label * v.weight;
        }

        //Update covariance(∑)
        foreach (v; fv) {
          cov[v.index] = 1.0
            / ((1.0 / cov[v.index]) + v.weight * v.weight / hyperparameter);
        }

        //Squared Hinge-loss
        return m * label < 0 ? 1 : 0;
      }
    

    /**
     * Predict
     * Params:
     *  fv =  feature
     * Returns: class label (+1, -1)
     */
    int Predict(feature[] fv)
      in
      {
        assert(fv != null);
      }
      out(result)
      {
        assert(result == -1 || result == +1);
      }
      body
      {
        double m = GetMargin(fv);
        return m > 0 ? 1 : -1;
      }


    feature[] ParseLine(string line, int label) {
      immutable string delim_value = ":";
      immutable string delim_cols = " ";

      feature[] fv;
      string[] columns = line.split(delim_cols);

      for(int i = 1; i < columns.length; i++) {
        string[] arr = columns[i].split(delim_value);

        feature f;

        if(arr.length != 2)
          continue;

        assert(arr != null);
        assert(arr.length == 2);

        f.index = to!int(arr[0]);
        f.weight = to!double(arr[1]);

        fv ~= f;
      }

      return fv;
    }


    example[] ReadData(string filename){
      Stream file = new BufferedFile(filename);
      size_t num_lines = 0;

      example[] data;

      foreach (char[] _line; file) {

        string line = cast(string)_line;

        if (line.length == 0) continue;
        if (line[0] == '#') continue;

        assert(line[0] == '-' || line[0] == '+');

        int label = line[0] == '+' ? +1 : -1;

        feature[] vec = ParseLine(line, label);

        if(vec != null) {
          example ex = new example();
          ex.label = label;
          ex.fv = vec;

          assert(vec != null);
          assert(ex.fv != null);
          assert(ex.label == -1 || ex.label == +1);

          data ~= ex;
        
        }
      }

      file.close();

      return data;
    }
}

void main(string[] args)
{
  version(all)
  {
    immutable uint dimention = 1355192;

    Arow arow = new Arow(dimention);
    example[] data = arow.ReadData("news20.binary");
   
    Random rand;
    randomShuffle(data, rand);
    ulong num_example = data.length;

    ulong train_size = cast(uint)(num_example * 0.75);
    ulong test_size = num_example - train_size;

    writefln("train: %d", cast(int)train_size);
    writefln("test: %d", cast(int)test_size);

    example[] train;
    example[] test;

    train.length = train_size;
    test.length = test_size;

    for(int i = 0; i < train_size; i++)
      train[i] = data[i];

    for(int i = 0; i < test_size; i++)
      test[i] = data[i + train_size];

    for(int i = 0; i < 3; i++) {
      //Train
      foreach(t; train) {
        arow.Update(t.fv, t.label);
      }

      //Predict
      int mistake = 0;
      foreach(t; test) {
        int label = arow.Predict(t.fv);
        if(label != t.label) mistake++;
      }

      writefln("%dth iteration:", i);
      writefln("Number of mistake: %d", mistake);
      writefln("Error rate: %f", mistake * 1.0 / test_size);
      writefln("");
    }
    
  }
}
