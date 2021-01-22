:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.

@ECHO OFF

rmdir source /S /Q
sphinx-apidoc -o .\source ..\backwardcompatibilityml
call make clean
call make html