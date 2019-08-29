import bisect
import re

class OffsetCache:
    def __init__(self):
        self.total_offset = 0
        self.offsets = []

    def add_delete(self,position,length):
        self.total_offset += length
        self.offsets.append((position - self.total_offset,self.total_offset))
    def resolve_offset(self,predicates):
        offest_positions = [ x[0] for x in self.offsets]

        for p in predicates:
            position_after_replace_deid = p["position"]
            index = bisect.bisect_right(offest_positions, p["position"])
            if index != 0:
                p["position"] = self.offsets[index-1][1] + position_after_replace_deid


class Replacer:

    def __init__(self,rules):
        self.rules = rules
        self.offset_caches = []

    def replace_text(self,text):
        result = text
        for (pattern,to_replace) in self.rules:
            to_replace_length = len(to_replace)
            offset_cache = OffsetCache()
            def add_offset(match):
                span = match.span()
                position  = span[1]
                length = (span[1] - span[0]) - to_replace_length
                offset_cache.add_delete(position,length)
                return to_replace
            result = re.sub(pattern,add_offset,result)
            self.offset_caches.append(offset_cache)
        return result

    def resolve_origin_offset(self,predicts):
        for offset_cache in reversed(self.offset_caches):
            offset_cache.resolve_offset(predicts)
        return predicts


# r = Replacer([
#     # deid pattern, the first  of tuple is the regular expression pattern, the second part of tuple is the text to replace
#     (r"\*\*DATE\*\*","x"), # I replace "**DATE**" with "x"
#     (r"\*\*NAME\*\*","y")
#
# ])

# origin_text = "**DATE** AB **NAME** AC **DATE** AE"
#
# # this will apply de id regular expression patterns and get the text after deid
# text_after_deid = r.replace_text(origin_text)
# print(text_after_deid)
#
# # this will resolve the original offset of the abbrs from the offset we get after deid
# origin_offsets = r.resolve_origin_offset([
#     {
#         "position":2,
#         "abbr":"AB",
#         "sense":""
#     },
#     {
#         "position":7,
#         "abbr":"AC",
#         "sense":""
#     },
#     {
#         "position":12,
#         "abbr":"AE",
#         "sense":""
#     }
# ])
#
# print(origin_offsets)
#
# # this valid the the original offset we resolved is correct
# for origin_offset in origin_offsets:
#     print(origin_text[origin_offset["position"]:origin_offset["position"] + len(origin_offset["abbr"])])
#
#
