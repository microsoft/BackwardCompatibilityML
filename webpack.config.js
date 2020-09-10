var webpack = require('webpack');

module.exports = {
  entry: ['./widget/index.tsx'],
  devtool: 'eval-source-map',
  performance: {
    hints: false
  },
  watch: false,
  plugins: [  
    new webpack.ProvidePlugin({
      $: "jquery",
      jQuery: "jquery"
    })
  ],
  module: {
    rules: [
        {
            test: /\.ts(x?)$/,
            exclude: /node_modules/,
            use: [
                {
                    loader: "ts-loader"
                }
            ]
        },
        // All output '.js' files will have any sourcemaps re-processed by 'source-map-loader'.
        {
            enforce: "pre",
            test: /\.js$/,
            loader: "source-map-loader"
        }
    ]
  },
  resolve: {
    extensions: ['*', '.js', '.jsx', '*.ts', '*.tsx']
  },
  output: {
    path: __dirname + '/backwardcompatibilityml/widget/resources',
    publicPath: '/',
    filename: 'widget-build.js'
  }
};
