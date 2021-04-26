#!/usr/bin/env python
# -*- coding: utf-8 -*-

#File: OCR_SROIE_1_0.py
#Version: 1.0
#Date: 2019-04-13
#Description: Evaluation script that computes Text Localization following the Deteval implementation (same as TL2p_deteval_1_0 with 4 points and Python 3 compatibility)

import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
import importlib
import json

def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """    
    return {
            }

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
                'GT_SAMPLE_NAME_2_ID':'(.+).txt',
                'DET_SAMPLE_NAME_2_ID':'(.+).txt',
                'CRLF':False
            }

def validate_data(gtFilePath, submFilePath,evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])

    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)
    
    
    if len(subm) != len(gt) :
        raise Exception("The Det file is not valid (invalid number of files in ZIP. Expected:" + str(len(gt)) + " Found:" + str(len(subm)) + ")")        
    
    #Validate format of GroundTruth
    for key in gt:
  
        gtJson = json.loads(gt[key])
        
        if not key in subm:
            raise Exception(("The Det file is not valid (file %s.txt not found)" % (key)).encode('utf-8', 'replace'))     
        
        
        if isinstance(gtJson, dict) == False :
            raise Exception(("The GT file is not valid (file %s.txt) The root element must be a dictionary." % (key)).encode('utf-8', 'replace'))     
        
        for intKey in gtJson:
            if not intKey in ['company','date','address','total']:
                raise Exception(("The GT file is not valid (file %s.txt) Key %s not valid (must be company,date,address or total)" % (key,intKey)).encode('utf-8', 'replace'))
        
        detJson = json.loads(subm[key])
        
        if isinstance(detJson, dict) == False :
            raise Exception(("The Det file is not valid (file %s.txt) The root element must be a dictionary." % (key)).encode('utf-8', 'replace'))     
        
        
        for intKey in detJson:
            if not intKey in ['company','date','address','total']:
                raise Exception(("The Det file is not valid (file %s.txt) Key %s not valid (must be company,date,address or total)" % (key,intKey)).encode('utf-8', 'replace'))

            if  isinstance(detJson[intKey], (dict,list,int,float)) == True:
                raise Exception(("The Det file is not valid (file %s.txt) Key %s must be an string" % (key,intKey)).encode('utf-8', 'replace'))     
        
    
def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """    

    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    
    perSampleMetrics = {}
    
    methodRecallSum = 0
    methodPrecisionSum = 0
    
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
   
    numGt = 0;
    numDet = 0;
   
    for key in gt:
        gtJson = json.loads(gt[key]);

        gtNumCategories  = len(gtJson.keys())
        gtCategories  = gtJson.keys()

        detNumCategories = 0
        
        recall = 0
        precision = 0
        hmean = 0        
        recallAccum = 0.
        precisionAccum = 0.
        
        log = ""
        
        if key in subm:
            detJson = json.loads(subm[key]);
            detNumCategories = len(detJson.keys())
 
            for detCategory,detValue in detJson.items():
                log += "<br>det category " + detCategory + " = <b>" + detValue + "</b> ";
                if detValue == gtJson[detCategory]:
                    log += "<span class='green'>correct</span>"
                    recallAccum +=1
                    precisionAccum +=1
                else:
                    log += "<span class='red'>incorrect</span>"
                    
        precision = float(0) if detNumCategories == 0 else float(precisionAccum) / detNumCategories
        recall = float(1) if gtNumCategories == 0 else float(recallAccum) / gtNumCategories
        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)  
            
        methodRecallSum += recallAccum
        methodPrecisionSum += precisionAccum
        numGt += gtNumCategories
        numDet += detNumCategories
        
        perSampleMetrics[key] = {
                                        'precision':precision,
                                        'recall':recall,
                                        'hmean':hmean,
                                        'gtCategories':gtNumCategories,
                                        'detCategories':detNumCategories,
                                        'correct':recallAccum,
                                        'log':log
                                    }
        
    methodRecall = 0 if numGt==0 else methodRecallSum/numGt
    methodPrecision = 0 if numDet==0 else methodPrecisionSum/numDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)
    
    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean  }

    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics}
    
    
    return resDict;



if __name__=='__main__':
        
    rrc_evaluation_funcs.main_evaluation(None,default_evaluation_params,validate_data,evaluate_method)

