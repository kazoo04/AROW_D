import std.stdio;
import std.math;
import std.file;
import std.stream;
import std.string;
import std.conv;
import std.random;

/***
 * ひとつの特徴を表す構造体
 */
struct feature {
  uint index;      ///特徴量のインデクス ( > 0 )
  double weight;  ///特徴量の値
}

/***
 * ひとつの学習データを表す構造体
 *
 * ひとつの学習データは、１つ以上の特徴と、教師データ(-1, +1)で構成されます
 */
class example {
  int label;      ///教師信号(-1, +1)
  feature[] fv;    ///特徴ベクトル
}

/***
 * Adaptive Regularization of Weight Vectors の実装
 *
 * See_Also:
 *   K. Crammer, A. Kulesza, and M. Dredze. "Adaptive regularization of weight vectors" NIPS 2009
 */
class Arow {
  private:
    size_t size;      /// 特徴の次元数N
    double[] mean;    /// 平均ベクトルμ
    double[] cov;     ///  分散行列∑ (リソース節約のために対角行列で近似)
    const double hyperparameter = 0.1; ///ハイパーパラメータr (r > 0)

    invariant()
    {
    }

  public:

    /**
     * コンストラクタ
     */
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
     * 認識識別面と特徴ベクトルfvの距離(マージン)を返す
     *
     * Returns: 識別超平面と特徴ベクトルfvのユークリッド距離
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

        /*
         * 内積を計算する
         * 識別するだけなら分散は考慮する必要がない
         */
        foreach(v; fv) {
          margin += mean[v.index] * v.weight;
        }

        return margin;
      }
    
    int CountZero()
    {
      int count;

      foreach(m; mean) {
        if(m == 0) {
          count++;
        }
      }

      writefln("zero = %f", count / cast(double)mean.length);

      return count;
    }


    /**
     * confidence(確信度)を計算して返す
     *
     * Returns: 確信度
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
        double confidence = 0.0;

        /*
         * confidenceの計算
         */
        foreach(v; fv) {
          confidence += cov[v.index] * v.weight * v.weight;
        }

        return confidence;
      }


    /**
     * 重みを更新する
     *
     * Returns: 損失 (0 or 1)
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
        //平均の更新
        foreach (v; fv) {
          mean[v.index] += alpha * cov[v.index] * label * v.weight;
        }

        //Update covariance(∑)
        //分散の更新
        foreach (v; fv) {
          cov[v.index] = 1.0
            / ((1.0 / cov[v.index]) + v.weight * v.weight / hyperparameter);
        }

        //損失の計算 (Squared Hinge-loss)
        //
        int loss = m * label < 0 ? 1 : 0;

        return loss;
      }
    

    /**
     * 認識結果を返す
     *
     * Returns: 認識結果(-1, +1)
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

    /**
     * 学習用データを1行受け取ってパースする
     */
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

    /**
     * トレーニングデータを読み込む
     */
    example[] ReadData(string filename){
      Stream file = new BufferedFile(filename);
      size_t num_lines = 0;

      example[] data;

      foreach (char[] _line; file) {

        string line = cast(string)_line;

        // 空行はスキップ
        if (line.length == 0) continue;

        // コメント行はスキップ
        if (line[0] == '#') continue;

        assert(line[0] == '-' || line[0] == '+');

        int label = line[0] == '+' ? +1 : -1;

        feature[] vec = ParseLine(line, label);

        // 読み込んだ特徴ベクトルを data に追加
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

    //const uint num_feature = 2;
    const uint dimention = 1355192;
    //uint num_example = 1355192;

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
      arow.CountZero();
      writefln("%dth iteration:", i);
      writefln("Number of mistake: %d", mistake);
      writefln("Error rate: %f", mistake * 1.0 / test_size);
      writefln("");
    }
    
  }
}
