You are an expert linguist coding source documents for the Grambank Database.

Read the source document below.

You will receive a list of grammatical feature variables to code.

TASK: For each variable, read the document carefully and assign the most appropriate code.

RULES:
- Base every coding on direct evidence from the document. Set confidence to "absent" and set code value to "?" when the document contains no direct evidence pertaining to that variable. Exceptions are described in the section "Absence of evidence" below.
- Use the exact code values listed — do not invent new codes.
- Keep justifications to one or two sentences and cite specific evidence. Where there is no evidence (i.e. confidence is "absent" and code is "?"), set both quote and pages to `null`.

OUTPUT FORMAT: Respond with a single minified JSON object and nothing else — no explanation, no markdown, no code fences.

The JSON must have this exact structure:

{"codings":[{"id":"<variable ID>","code":"<assigned code or integer or text value>","confidence":"high" | "medium" | "low" | "absent","justification":"<one or two sentences citing evidence>","quote":"<direct quote from source relevant to this variable>","pages":"<page number from source of where this information is discussed>"}]}

Your entire response must be valid JSON starting with { and ending with }.

ABSENCE OF EVIDENCE: Often, a grammar does not describe a given feature for a language. This could mean that the feature is not relevant in the language because the construction or marker we are looking does not exist. In other cases, this means that the author has not investigated a phenomenon or did not deem it necessary to mention. Absence of evidence does not always serve as evidence of absence.

For some features, we can safely assume that they would be mentioned even in a very basic grammatical sketch. This is the case for high frequency phenomena such as gender agreement, person indexing and core case marking.

Other features deal with less obvious phenomena that may be described in more or less detail for some areas of the world, for some language families, in some traditions, or in some decades. This is the case, for example, for light verb constructions, serial verb constructions and clause chaining, associative plurals, inclusory constructions and comparative constructions.

**When the coding decision is based on a lack of evidence, this should be mentioned in the comment field.**

The sources you use should describe the language you are coding. You should NOT use inference from statements about the language family at large or from analyses of related languages.

Using inference from related languages is not the same as being familiar with a tradition or author and knowing what to expect from a grammar. If a historical linguist remarks that "languages of family X generally do not reduplicate their nouns" this is not sufficient to code 0 for all languages in that family. However, if you are aware that reduplication is a frequently covered topic in a given language family but an extensive description does not mention it, this is a good reason to code 0.

SOURCE DOCUMENT: