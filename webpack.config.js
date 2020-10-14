const webpack = require('webpack');
const path = require('path');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

const jupyterConfig = {
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
    }),
    new MiniCssExtractPlugin({
      filename: 'widget.css'
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
        },
        {
            test: /\.css$/,
            use: [
              MiniCssExtractPlugin.loader,
              'css-loader'
            ]
        }
    ]
  },
  resolve: {
    extensions: ['*', '.js', '.jsx', '*.ts', '*.tsx']
  },
  output: {
    path: path.resolve(__dirname, 'backwardcompatibilityml/widget/resources'),
    publicPath: '/',
    filename: 'widget-build.js'
  }
};

const devConfig = {
  entry: [
    'react-hot-loader/patch',
    './widget/index.tsx'
  ],
  mode: 'development',
  devtool: 'eval-source-map',
  performance: {
    hints: false
  },
  plugins: [  
    new webpack.ProvidePlugin({
      $: "jquery",
      jQuery: "jquery"
    }),
    new webpack.HotModuleReplacementPlugin()
  ],
  module: {
    rules: [
      {
        test: /\.ts(x)?$/,
        loader: 'ts-loader',
        exclude: /node_modules/
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
            'style-loader',
            'css-loader'
          ]
      }
    ]
  },
  resolve: {
    extensions: ['*', '.js', '.jsx', '*.ts', '*.tsx'],
    alias: {
      'react-dom': '@hot-loader/react-dom'
    }
  },
  output: {
    path: path.resolve(__dirname, 'dev-artifacts'),
    publicPath: '/dev-artifacts/',
    filename: 'widget-build-dev.js'
  },
  devServer: {
    contentBase: path.resolve(__dirname, 'dev-artifacts'),
    port: 3000,
    publicPath: 'http://localhost:3000/dev-artifacts/',
    hotOnly: true
  },
};

// order is important -- webpack-dev-server will use the first exported config
module.exports = [ devConfig, jupyterConfig ]
