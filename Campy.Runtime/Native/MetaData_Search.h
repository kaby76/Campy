#pragma once

function_space_specifier U32 MetaData_CompareNameAndSig(STRING name, BLOB_ sigBlob, tMetaData *pSigMetaData, tMD_TypeDef **ppSigClassTypeArgs, tMD_TypeDef **ppSigMethodTypeArgs, tMD_MethodDef *pMethod, tMD_TypeDef **ppMethodClassTypeArgs, tMD_TypeDef **ppMethodMethodTypeArgs);
function_space_specifier tMD_FieldDef* MetaData_FindFieldInType(tMD_TypeDef *pTypeDef, STRING name);
function_space_specifier tMD_FieldDef* MetaData_FindFieldInTypeAll(tMD_TypeDef *pTypeDef, STRING name);
function_space_specifier tMetaData* MetaData_GetResolutionScopeMetaData(tMetaData *pMetaData, IDX_TABLE resolutionScopeToken, tMD_TypeDef **ppInNestedType);
function_space_specifier tMD_TypeDef* MetaData_GetTypeDefFromName(tMetaData *pMetaData, STRING nameSpace, STRING name, tMD_TypeDef *pInNestedClass);
function_space_specifier tMD_TypeDef* MetaData_GetTypeDefFromFullName(STRING assemblyName, STRING nameSpace, STRING name);
function_space_specifier tMD_TypeDef* MetaData_GetTypeDefFromFullNameAndNestedType(STRING assemblyName, STRING nameSpace, STRING name, tMD_TypeDef* nested);
function_space_specifier tMD_TypeDef* MetaData_GetTypeDefFromDefRefOrSpec(tMetaData *pMetaData, IDX_TABLE token, tMD_TypeDef **ppClassTypeArgs, tMD_TypeDef **ppMethodTypeArgs);
function_space_specifier tMD_TypeDef* MetaData_GetTypeDefFromMethodDef(tMD_MethodDef *pMethodDef);
function_space_specifier tMD_TypeDef* MetaData_GetTypeDefFromFieldDef(tMD_FieldDef *pFieldDef);
function_space_specifier tMD_MethodDef* MetaData_GetMethodDefFromDefRefOrSpec(tMetaData *pMetaData, IDX_TABLE token, tMD_TypeDef **ppClassTypeArgs, tMD_TypeDef **ppMethodTypeArgs);
function_space_specifier tMD_FieldDef* MetaData_GetFieldDefFromDefOrRef(tMetaData *pMetaData, IDX_TABLE token, tMD_TypeDef **ppClassTypeArgs, tMD_TypeDef **ppMethodTypeArgs);
function_space_specifier PTR MetaData_GetTypeMethodField(tMetaData *pMetaData, IDX_TABLE token, U32 *pObjectType, tMD_TypeDef **ppClassTypeArgs, tMD_TypeDef **ppMethodTypeArgs);
function_space_specifier tMD_ImplMap* MetaData_GetImplMap(tMetaData *pMetaData, IDX_TABLE memberForwardedToken);
function_space_specifier STRING MetaData_GetModuleRefName(tMetaData *pMetaData, IDX_TABLE memberRefToken);