import CopyWebpackPlugin from 'copy-webpack-plugin';
import TerserPlugin from 'terser-webpack-plugin';
import { fileURLToPath } from 'url';
import path from 'path';
import { PyodidePlugin } from "@pyodide/webpack-plugin";
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * @type {import('webpack').Configuration}
 */
export default {
    mode: 'development',
    devtool: 'source-map',
    entry: {
        'dist/index': './index.js',
        'dist/index.min': './index.js',
    },
    output: {
        filename: '[name].js',
        path: __dirname,
        library: {
            type: 'module',
        },
    },
    plugins: [
        // Copy .wasm files to dist folder
        new CopyWebpackPlugin({
            patterns: [
                {
                    from: 'node_modules/onnxruntime-web/dist/*.wasm',
                    to: 'dist/[name][ext]'
                },
            ],
        }),
        new PyodidePlugin(),
    ],
    optimization: {
        minimize: true,
        minimizer: [new TerserPlugin({
            test: /\.min\.js$/,
            extractComments: false,
        })],
    },
    devServer: {
        static: {
            directory: __dirname
        },
        port: 8080,
        https: {
            key: fs.readFileSync('key.pem'),
            cert: fs.readFileSync('cert.pem'),
        },
    },
    experiments: {
        outputModule: true,
    }
};
