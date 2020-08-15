# -*- coding: utf-8 -*


import sys 
import numpy as np 
import pandas as pd 
import json
import warnings
from scipy import stats
from itertools import combinations, groupby
from collections import Counter
from data_feature import feature_newFeature, feature_numCateCols, feature_concentratedCols, feature_numColBinning, feature_cateColCombineLevel
warnings.filterwarnings('ignore')





class Apriori():
	''' 
	discretize numerical variables to categorical variables
	transform DataFrame to transactonal data set
	mining association rules using Apriori alogrithme take advantage of iteration and generation in python 
	'''

	def __init__(self, df, tag_name, num_cols, cate_cols, target_cols, num_entropyTreshold, cate_entropyThreshold, p_label, label, numToCateBins, numToCateLabels, numToCateMethod, numToCateQuantiles, highCateLevel_threshold, support_threshold, max_freqItemCt, consequent, confidence_threshold, lift_threshold, consequent_method, sort_rulse_by='lift'):
		''' initialization '''
		self.self = self
		self.df = df
		self.tag_name = tag_name
		self.num_cols = num_cols
		self.cate_cols = cate_cols
		self.target_cols = target_cols
		self.num_entropyTreshold = num_entropyTreshold
		self.cate_entropyThreshold = cate_entropyThreshold
		self.p_label = p_label
		self.label = label
		self.numToCateBins = numToCateBins
		self.numToCateLabels = numToCateLabels
		self.numToCateMethod = numToCateMethod
		self.numToCateQuantiles = numToCateQuantiles
		self.highCateLevel_threshold = highCateLevel_threshold
		self.support_threshold = support_threshold
		self.max_freqItemCt = max_freqItemCt
		self.consequent = consequent
		self.confidence_threshold = confidence_threshold
		self.lift_threshold = lift_threshold
		self.consequent_method = consequent_method
		self.sort_rulse_by = sort_rulse_by
		return 


	def run(self):
		''' run '''
		# get frequency items
		order_nums, freq_items, cateCol_otherLevels = self.apriori_getFreqItems(self.df, self.tag_name, self.num_cols, self.cate_cols, self.target_cols, self.num_entropyTreshold, self.cate_entropyThreshold, self.p_label, self.label, self.numToCateBins, self.numToCateLabels, self.numToCateMethod, self.numToCateQuantiles, self.highCateLevel_threshold, self.support_threshold, self.max_freqItemCt, self.consequent)
		# get association rules
		rules = self.apriori_getRules(order_nums, freq_items, self.confidence_threshold, self.lift_threshold, self.consequent, self.consequent_method)
		# sort association rules
		result = self.apriori_sortRules(rules, self.sort_rulse_by)
		# print and save rules
		self.apriori_printSaveRules(result)
		return


	def apriori_delCols(self, df, num_cols, cate_cols, target_cols, consequent):
		''' maintain the target variables:taget_cols，delete other variables '''
		if consequent is not None and consequent.split('::')[0] not in target_cols:
		    target_cols.append(consequent.split('::')[0]) 
		for num_col in num_cols:
		    if num_col not in target_cols:
		        del df[num_col]
		        num_cols.remove(num_col)
		for cate_col in cate_cols:
		    if cate_col not in target_cols:
		        del df[cate_col]
		        cate_cols.remove(cate_col)				
		return


	def apriori_concentratedCols(self, df, num_cols, cate_cols, num_entropyTreshold, cate_entropyThreshold, consequent):
		'''
		del ove concentrated numerical and categorical variables
		if consequent is not null, then consequent would be the rule consequent
		'''
		if consequent is not None:
		    if consequent in num_cols:
		        num_entropyTreshold = stats.entropy(pk=df[consequent.split('::')[0]].value_counts(normalize=True), base=2)
		    elif consequent in cate_cols:
		        cate_entropyThreshold = stats.entropy(pk=df[consequent.split('::')[0]].value_counts(normalize=True), base=2)
		concentratedCols = feature_concentratedCols(df, num_cols, cate_cols, num_entropyTreshold, cate_entropyThreshold)
		for num_concentratedCol in concentratedCols['num_concentratedCols']:
		    del df[num_concentratedCol]
		    num_cols.remove(num_concentratedCol)
		for cate_concentratedCol in concentratedCols['cate_concentratedCols']:
		    del df[cate_concentratedCol]
		    cate_cols.remove(cate_concentratedCol)
		return


	def apriori_highCateLevel(self, df, cate_cols, highCateLevel_threshold):
		''' label categorical level as nan which accounts for over high frequency '''
		for cate_col in cate_cols:
		    vc = df[cate_col].value_counts(normalize=True)
		    top_level = vc.index[0]
		    if vc.values[0] > highCateLevel_threshold:
		        df[cate_col][df[cate_col] == top_level] = np.nan
		return


	def apriori_dfFormat(self, df):
		''' transform DataFrame shape to transaction data set '''
		df = df.stack().reset_index()
		df.rename(columns={'level_0':'order_id', 'level_1':'item_id1', 0:'item_id2'}, inplace=True)
		df['item_id'] = df['item_id1'].apply(lambda x: str(x)) + '::' + df['item_id2'].apply(lambda x: str(x))
		del df['item_id1']
		del df['item_id2']
		return df


	def apriori_genCombinations(self, df, combine_num):
		''' generate goods combination taking advantage of generation '''
		for order_id, item_id in groupby(df.values, lambda x:x[0]):
		    item_list = [item[1] for item in item_id]
		    for item_combine in combinations(item_list, combine_num):
		        yield (item_combine)


	def apriori_genCombinationsOrderid(self, df, order_id, combine_num):
		''' generate order id combination taking advantage of generation '''
		item_ids = df['item_id'][df['order_id'] == order_id]
		for item_combine in combinations(item_ids, combine_num):
		    item_combine = list(item_combine)
		    item_combine.sort()
		    item_combine = '##'.join(item_combine)
		    yield item_combine


	def apriori_genCombinationsOrderid2(self, series, combine_num):
		''' generate order id combination taking advantage of generation: another way '''
		for item_combine in combinations(series, combine_num):
		    item_combine = list(item_combine)
		    item_combine.sort()
		    item_combine = '##'.join(item_combine)
		    yield item_combine


	def apriori_getFreqItemsSub(self, df, min_support, max_freqItemCt):
		''' get frequency items: method 1 '''
		print('----------------------------------------------------------------------------------')
		print('apriori_getFreqItems: start')
		# apriori
		freq_items = {}
		# one frequency items
		item_ct = df['item_id'].value_counts()
		item_ct = item_ct[item_ct >= min_support]
		freq_items_one = {}
		for i in range(item_ct.size):
		    freq_items_one[item_ct.index[i]] = item_ct.values[i]
		freq_items[1] = freq_items_one
		df = df[df['item_id'].isin(item_ct.index)]
		# mulit frequency items
		combine_num = 2
		while df.shape[0] > 0 and combine_num <= max_freqItemCt:
		    # gen mulit items from df
		    gen_combinations = self.apriori_genCombinations(df, combine_num)
		    item_combines = pd.Series(Counter(gen_combinations)).rename('freq')
		    item_combines = item_combines.to_frame().reset_index()
		    # get multi frequency items from mulit items
		    item_combines.query("freq > {0}".format(min_support), inplace=True)
		    if item_combines.shape[0] > 0:
		        level_cols = []
		        for i in range(combine_num):
		            level_cols.append('_'.join(['level', str(i)]))
		        item_arrays = []
		        for level_col in level_cols:
		            item_arrays.append(item_combines[level_col])
		        item_arrays_temp = []
		        zip_item_arrays = zip(*item_arrays)
		        for zip_item in zip_item_arrays:
		            zip_item = list(zip_item)
		            zip_item.sort()
		            item_arrays_temp.append('##'.join(zip_item))
		        freq_items_here = {}
		        freqs = item_combines['freq']
		        for i in range(item_combines.shape[0]):
		            freq_items_here[item_arrays_temp[i]] = freqs.iloc[i]
		        freq_items[combine_num] = freq_items_here
		        print(combine_num)
		        # on condition of each order_id in df, gen their combinations, and filter items not in frequency items
		        order_ids = df['order_id'].unique()
		        for order_id in order_ids:
		            item_set = []
		            gen_combinationsOrderid = self.apriori_genCombinationsOrderid(df, order_id, combine_num)
		            for item_combine in gen_combinationsOrderid:
		                if item_combine in freq_items_here:
		                    item_set.extend(item_combine.split('##'))
		                    item_set = list(set(item_set))        
		            drop_index = df.index[(df['order_id'] == order_id) & (-df['item_id'].isin(item_set))]
		            df.drop(index=drop_index, inplace=True)
		    # filter orders with less than combine_num goods
		    order_ct = df['order_id'].value_counts()
		    maintain_orders = order_ct[order_ct > combine_num].index
		    df = df[df['order_id'].isin(maintain_orders)]
		    combine_num += 1
		print('apriori_getFreqItems: finish')
		print('----------------------------------------------------------------------------------')
		return freq_items


	def apriori_getFreqItemsSub3(self, df, min_support, max_freqItemCt):
		''' get frequency items: method 3 ''' 
		print('----------------------------------------------------------------------------------')
		print('apriori_getFreqItems: start')
		# apriori
		freq_items = {}
		# one frequency items
		item_ct = df['item_id'].value_counts()
		item_ct = item_ct[item_ct >= min_support]
		freq_items_one = {}
		for i in range(item_ct.size):
		    freq_items_one[item_ct.index[i]] = item_ct.values[i]
		freq_items[1] = freq_items_one
		df = df[df['item_id'].isin(item_ct.index)]
		# mulit frequency items
		combine_num = 2
		while df.shape[0] > 0 and combine_num <= max_freqItemCt:
		    # gen mulit items from df
		    gen_combinations = self.apriori_genCombinations(df, combine_num)
		    item_combines = pd.Series(Counter(gen_combinations)).rename('freq')
		    item_combines = item_combines.to_frame().reset_index()
		    # get multi frequency items from mulit items
		    item_combines.query("freq > {0}".format(min_support), inplace=True)
		    if item_combines.shape[0] > 0:
		        level_cols = []
		        for i in range(combine_num):
		            level_cols.append('_'.join(['level', str(i)]))
		        item_arrays = []
		        for level_col in level_cols:
		            item_arrays.append(item_combines[level_col])
		        item_arrays_temp = []
		        zip_item_arrays = zip(*item_arrays)
		        for zip_item in zip_item_arrays:
		            zip_item = list(zip_item)
		            zip_item.sort()
		            item_arrays_temp.append('##'.join(zip_item))
		        freq_items_here = {}
		        freqs = item_combines['freq']
		        for i in range(item_combines.shape[0]):
		            freq_items_here[item_arrays_temp[i]] = freqs.iloc[i]
		        freq_items[combine_num] = freq_items_here
		        print(combine_num)
		    # filter orders with less than combine_num goods
		    order_ct = df['order_id'].value_counts()
		    maintain_orders = order_ct[order_ct > combine_num].index
		    df = df[df['order_id'].isin(maintain_orders)]
		    combine_num += 1
		print('apriori_getFreqItems: finish')
		print('----------------------------------------------------------------------------------')
		return freq_items


	def apriori_getFreqItemsSub2Sub(self, group_df, drop_index, combine_num, freq_items_here):
		''' assisting function for apriori_getFreqItemsSub2 '''
		item_set = []
		gen_combinationsOrderid = self.apriori_genCombinationsOrderid2(group_df['item_id'], combine_num)
		for item_combine in gen_combinationsOrderid:
		    item_set.append(item_combine)
		    if item_combine in freq_items_here:
		        item_set.extend(item_combine.split('##'))
		        item_set = list(set(item_set))
		drop_index.extend(group_df['item_id'].index[-(group_df['item_id'].isin(item_set))])
		return None


	def apriori_getFreqItemsSub2(self, df, min_support, max_freqItemCt):
		''' get frequency items: method 2 '''
		print('----------------------------------------------------------------------------------')
		print('apriori_getFreqItems: start')
		# apriori
		freq_items = {}
		# one frequency items
		item_ct = df['item_id'].value_counts()
		item_ct = item_ct[item_ct >= min_support]
		freq_items_one = {}
		for i in range(item_ct.size):
		    freq_items_one[item_ct.index[i]] = item_ct.values[i]
		freq_items[1] = freq_items_one
		df = df[df['item_id'].isin(item_ct.index)]
		# mulit frequency items
		combine_num = 2
		while df.shape[0] > 0 and combine_num <= max_freqItemCt:
		    # gen mulit items from df
		    gen_combinations = self.apriori_genCombinations(df, combine_num)
		    item_combines = pd.Series(Counter(gen_combinations)).rename('freq')
		    item_combines = item_combines.to_frame().reset_index()
		    # get multi frequency items from mulit items
		    item_combines.query("freq > {0}".format(min_support), inplace=True)
		    if item_combines.shape[0] > 0:
		        level_cols = []
		        for i in range(combine_num):
		            level_cols.append('_'.join(['level', str(i)]))
		        item_arrays = []
		        for level_col in level_cols:
		            item_arrays.append(item_combines[level_col])
		        item_arrays_temp = []
		        zip_item_arrays = zip(*item_arrays)
		        for zip_item in zip_item_arrays:
		            zip_item = list(zip_item)
		            zip_item.sort()
		            item_arrays_temp.append('##'.join(zip_item))
		        freq_items_here = {}
		        freqs = item_combines['freq']
		        for i in range(item_combines.shape[0]):
		            freq_items_here[item_arrays_temp[i]] = freqs.iloc[i]
		        freq_items[combine_num] = freq_items_here
		        print(combine_num)
		        # on condition of each order_id in df, gen their combinations, and filter items not in frequency items
		        drop_index = []
		        _ = df.groupby('order_id').apply(self.apriori_getFreqItemsSub2Sub, drop_index, combine_num, freq_items_here)
		        df.drop(index=drop_index, inplace=True)
		    # filter orders with less than combine_num goods
		    order_ct = df['order_id'].value_counts()
		    maintain_orders = order_ct[order_ct > combine_num].index
		    df = df[df['order_id'].isin(maintain_orders)]
		    combine_num += 1
		print('apriori_getFreqItems: finish')
		print('----------------------------------------------------------------------------------')
		return freq_items


	def apriori_unsupervisedRulesFromConfidenceLift(self, order_nums, freq_items, confidence_threshold, lift_threshold, rules):
		''' get rules from frequency items according to confidence and lift '''
		print('----------------------------------------------------------------------------------')
		print('apriori_unsupervisedRulesFromConfidence: start')
		for combine_num, freq_item_kv in freq_items.items():         
		    if combine_num > 1:
		        print(combine_num)
		        for freq_item, n_freqItem in freq_item_kv.items():
		            support = n_freqItem / order_nums
		            freq_item = set(freq_item.split('##'))
		            m = len(freq_item)
		            i = m - 1
		            while i >= 1:
		                for antecedent in combinations(freq_item, i):
		                    consequent = list(freq_item - set(antecedent))
		                    consequent.sort()
		                    consequent = '##'.join(consequent)
		                    antecedent = list(antecedent)
		                    antecedent.sort()
		                    antecedent = '##'.join(antecedent)
		                    confidence = n_freqItem / freq_items[i][antecedent]
		                    lift = (order_nums * n_freqItem) / (freq_items[i][antecedent] * freq_items[combine_num - i][consequent])
		                    if confidence_threshold is not None and lift_threshold is None and confidence > confidence_threshold:
		                        rules['-->'.join([antecedent, consequent])] = {'support':support, 'confidence':confidence}
		                    elif confidence_threshold is None and lift_threshold is not None and lift > lift_threshold:
		                        rules['-->'.join([antecedent, consequent])] = {'support':support, 'lift':lift}
		                    elif confidence_threshold is not None and lift_threshold is not None and confidence > confidence_threshold and lift > lift_threshold:
		                        rules['-->'.join([antecedent, consequent])] = {'support':support, 'confidence':confidence, 'lift':lift}
		                i -= 1
		print('apriori_unsupervisedRulesFromConfidence: finish')
		print('----------------------------------------------------------------------------------')
		return


	def apriori_supervisedRulesFromConfidenceLift(self, order_nums, freq_items, confidence_threshold, lift_threshold, rules, consequent):
		''' get rules from frequency items according to confidence and lift, and the rule consequent contains conseqent '''
		print('----------------------------------------------------------------------------------')
		print('apriori_supervisedRulesFromConfidence: start')
		for combine_num, freq_item_kv in freq_items.items():
		    if combine_num > 1:
		        print(combine_num)
		        for freq_item, n_freqItem in freq_item_kv.items():
		            support = n_freqItem / order_nums
		            freq_item = set(freq_item.split('##'))
		            if consequent in freq_item:
		                freq_item.remove(consequent)
		                m = len(freq_item)
		                i = m 
		                while i >= 1:
		                    for antecedent in combinations(freq_item, i):
		                        antecedent = list(antecedent)
		                        antecedent.sort()
		                        antecedent = '##'.join(antecedent)
		                        confidence = n_freqItem / freq_items[i][antecedent]
		                        confidence = n_freqItem / freq_items[i][antecedent]
		                        lift = (order_nums * n_freqItem) / (freq_items[i][antecedent] * freq_items[1][consequent])
		                        if confidence_threshold is not None and lift_threshold is None and confidence > confidence_threshold:
		                            rules['-->'.join([antecedent, consequent])] = {'support':support, 'confidence':confidence}
		                        elif confidence_threshold is None and lift_threshold is not None and lift > lift_threshold:
		                            rules['-->'.join([antecedent, consequent])] = {'support':support, 'lift':lift}
		                        elif confidence_threshold is not None and lift_threshold is not None and confidence > confidence_threshold and lift > lift_threshold:
		                            rules['-->'.join([antecedent, consequent])] = {'support':support, 'confidence':confidence, 'lift':lift}
		                    i -= 1                                                    
		print('apriori_supervisedRulesFromConfidence: finish')
		print('----------------------------------------------------------------------------------')
		return


	def apriori_containRulesFromConfidenceLift(self, order_nums, freq_items, confidence_threshold, lift_threshold, rules, consequent):
		''' get rules from frequency items according to confidence and lift, and the pre-rule key contains 'consequent' '''
		print('----------------------------------------------------------------------------------')
		print('model_apriori_containRulesFromConfidenceLift: start')
		for combine_num, freq_item_kv in freq_items.items():         
		    if combine_num > 1:
		        print(combine_num)
		        for freq_item, n_freqItem in freq_item_kv.items():
		            support = n_freqItem / order_nums
		            freq_item = set(freq_item.split('##'))
		            if consequent in freq_item:
		                m = len(freq_item)
		                i = m - 1
		                while i >= 1:
		                    for antecedent in combinations(freq_item, i):
		                        consequent = list(freq_item - set(antecedent))
		                        consequent.sort()
		                        consequent = '##'.join(consequent)
		                        antecedent = list(antecedent)
		                        antecedent.sort()
		                        antecedent = '##'.join(antecedent)
		                        confidence = n_freqItem / freq_items[i][antecedent]
		                        lift = (order_nums * n_freqItem) / (freq_items[i][antecedent] * freq_items[combine_num - i][consequent])
		                        if confidence_threshold is not None and lift_threshold is None and confidence > confidence_threshold:
		                            rules['-->'.join([antecedent, consequent])] = {'support':support, 'confidence':confidence}
		                        elif confidence_threshold is None and lift_threshold is not None and lift > lift_threshold:
		                            rules['-->'.join([antecedent, consequent])] = {'support':support, 'lift':lift}
		                        elif confidence_threshold is not None and lift_threshold is not None and confidence > confidence_threshold and lift > lift_threshold:
		                            rules['-->'.join([antecedent, consequent])] = {'support':support, 'confidence':confidence, 'lift':lift}
		                    i -= 1
		print('apriori_containRulesFromConfidenceLift: finish')
		print('----------------------------------------------------------------------------------')
		return


	def apriori_getFreqItems(self, df, tag_name, num_cols, cate_cols, target_cols, num_entropyTreshold, cate_entropyThreshold, p_label, label, numToCateBins, numToCateLabels, numToCateMethod, numToCateQuantiles, highCateLevel_threshold, support_threshold, max_freqItemCt, consequent):
		''' main process of getting freq items '''
		df = df.query(tag_name)
		order_nums = df.shape[0]
		min_support = support_threshold * order_nums
		# del unnecessary varialbes
		self.apriori_delCols(df, num_cols, cate_cols, target_cols, consequent)
		# del over concentrated variables
		self.apriori_concentratedCols(df, num_cols, cate_cols, num_entropyTreshold, cate_entropyThreshold, consequent)
		# merge categorical varialbe levels which accounts for low frequency
		cateCol_otherLevels = {}
		for cate_col in cate_cols:
		    other_levels = feature_cateColCombineLevel(df, cate_col, p_label, label)
		    cateCol_otherLevels[cate_col] = other_levels
		# binning for numerical varialbes
		temp_cols = [col for col in num_cols]
		for num_col in temp_cols:
		    feature_numColBinning(df, num_cols, cate_cols, num_col, numToCateBins, numToCateLabels, numToCateMethod, numToCateQuantiles)
		# label categorical level as nan which accounts for over high frequency 
		self.apriori_highCateLevel(df, cate_cols, highCateLevel_threshold)
		# transform DataFrame shape to transaction data set
		df = self.apriori_dfFormat(df)
		# get freq items according to support
		if consequent is not None:
		    min_support = min(min_support, (df['item_id'] == consequent).sum())
		freq_items = self.apriori_getFreqItemsSub3(df, min_support, max_freqItemCt)
		return order_nums, freq_items, cateCol_otherLevels


	def apriori_getRules(self, order_nums, freq_items, confidence_threshold, lift_threshold, consequent, consequent_method):
		''' get association rules from freq items '''
		rules = {}
		# association rules
		if consequent is None:
		    self.apriori_unsupervisedRulesFromConfidenceLift(order_nums, freq_items, confidence_threshold, lift_threshold, rules)
		elif consequent_method == 'consequent':
		    self.apriori_supervisedRulesFromConfidenceLift(order_nums, freq_items, confidence_threshold, lift_threshold, rules, consequent)
		elif consequent_method == 'contain':
		    self.apriori_containRulesFromConfidenceLift(order_nums, freq_items, confidence_threshold, lift_threshold, rules, consequent)
		return rules


	def apriori_sortRules(self, rules, by):
		''' sort rules order by lift or confidence '''
		result = [(k, v) for k, v in rules.items()]
		result.sort(key=lambda x:x[1][by], reverse=True)
		return result


	def apriori_printSaveRules(self, result):
		''' print and save rules '''
		for rule in result:
		    print(rule[0])
		    print('\t\t%s\n' % rule[1])
		with open('apriori.json', 'w', encoding='utf-8') as f:
		    json.dump(result, f, ensure_ascii=False, indent=4)








