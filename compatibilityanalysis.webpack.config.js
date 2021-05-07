const webpack = require('webpack');
const path = require('path');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const ReactRefreshPlugin = require('@pmmmwh/react-refresh-webpack-plugin');

const isDevelopment = process.env.NODE_ENV !== 'production';

module.exports = {
  mode: isDevelopment ? 'development' : 'production',
  entry: ['./widgets/compatibilityanalysis/index.tsx'],
  devtool: 'eval-source-map',
  performance: {
    hints: false
  },
  watch: false,
  plugins: [  
    new webpack.ProvidePlugin({
      $: "jquery",
      jQuery: "jquery"
    }),
    new MiniCssExtractPlugin({
      filename: 'widget.css'
    }),
    isDevelopment && new HtmlWebpackPlugin({
      template: './development/compatibility-analysis/index.html'
    }),
    isDevelopment && new ReactRefreshPlugin()
  ].filter(Boolean),
  module: {
    rules: [
      {
        test: /\.ts(x?)$/,
        exclude: /node_modules/,
        use: [
          isDevelopment && {
            loader: 'babel-loader',
            options: { plugins: ['react-refresh/babel'] }
          },
          {
              loader: "ts-loader"
          }
        ].filter(Boolean)
      },
      // All output '.js' files will have any sourcemaps re-processed by 'source-map-loader'.
      {
        enforce: "pre",
        test: /\.js$/,
        loader: "source-map-loader"
      },
      {
        test: /\.css$/,
        use: [
          isDevelopment && 'style-loader',
          !isDevelopment && MiniCssExtractPlugin.loader,
          'css-loader'
        ].filter(Boolean)
      },
      {
        test: /\.(png|jpe?g|gif)$/i,
        use: [
          {
            loader: 'file-loader',
          },
        ],
      }
    ]
  },
  resolve: {
    extensions: ['*', '.js', '.jsx', '*.ts', '*.tsx']
  },
  output: {
    path: path.resolve(__dirname, 'backwardcompatibilityml/widgets/compatibility_analysis/resources'),
    publicPath: '/',
    filename: 'widget-build.js'
  },
  devServer: {
    contentBase: path.resolve(__dirname, "development/compatibility-analysis"),
    port: 3000
  }
};
