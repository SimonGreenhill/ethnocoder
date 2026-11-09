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

CODING HINTS: There follows general advice to help with coding. The source document is at the end.

LOGICAL DEPENDENCIES: There are a few groups of features with logical dependencies of different kinds. E.g. a Yes for either of GB022 or GB023 (about articles of different kinds) will almost certainly be accompanied by a Yes for at least one of GB020 or GB021 (about whether there are definite and indefinite articles, respectively). At least one of GB131, GB132 and GB133 (word order) must be Yes, as must at least one of GB408, GB409 and GB410 (alignment system).

ABSENCE OF EVIDENCE: Often, a grammar does not describe a given feature for a language. This could mean that the feature is not relevant in the language because the construction or marker we are looking does not exist. In other cases, this means that the author has not investigated a phenomenon or did not deem it necessary to mention. Absence of evidence does not always serve as evidence of absence.

For some features, we can safely assume that they would be mentioned even in a very basic grammatical sketch. This is the case for high frequency phenomena such as **gender agreement**, **person indexing** and **core case marking**.

Other features deal with less obvious phenomena that may be described in more or less detail for some areas of the world, for some language families, in some traditions, or in some decades. This is the case, for example, for **light verb constructions**, **serial verb constructions** and **clause chaining**, **associative plurals**, **inclusory constructions** and **comparative constructions**.

**When the coding decision is based on a lack of evidence, this should be mentioned in the comment field.**

The sources you use should describe the language you are coding. You should NOT use inference from statements about the language family at large or from analyses of related languages.

Using inference from related languages is not the same as being familiar with a tradition or author and knowing what to expect from a grammar. If a historical linguist remarks that "languages of family X generally do not reduplicate their nouns" this is not sufficient to code 0 for all languages in that family. However, if you are aware that reduplication is a frequently covered topic in a given language family but an extensive description does not mention it, this is a good reason to code 0.

LANGUAGE CHANGE: In cases where there is a variation that arises because of a change in progress, code for the older state.

BORROWING: Where a strategy has been borrowed recently, and the borrowing process is unmistakable, we are only interested in the more archaic strategies. Mention the recent borrowing in the comment field.

WORDHOOD AND PHONOLOGICAL BOUNDNESS: Boundedness is phonological boundedness. A morpheme is phonologically free if it occurs as a phonological word. It is phonologically bound if it does not occur as its own phonological word. 

Criteria on boundedness are:
  
  1. **stress**: if a morpheme's stress depends on other morphemes it occurs with or if it falls within another morpheme's stress domain and cannot be stressed by itself, it is phonologically bound.
  2. **boundary phenomena** or **(phonological) allomorphy**: if a morpheme's segmental form depends on other morphemes it occurs with, it is phonologically bound.
  3. **vowel harmony**: if a morpheme is affected by or causes vowel harmony, it is phonologically bound.

In practice, grammars rarely indicate this. So, we use **orthographic boundedness** instead. If a form is orthographically bound, it is assumed to be phonologically bound. If it stands on its own orthographically, it counts as a separate word.

MORPHOLOGY: Features GB047, GB070, GB082, GB103, GB107, GB115 aim to capture marking by means of affixes and clitics as well as suppletion, reduplication and tone, unless this is stated otherwise in the feature description. For the purpose of Grambank, morphological marking does not include serial verb-constructions, verb compounding or clause-chaining, or marking by phonologically independent means (such as particles or auxiliaries).

AFFIXES AND CLITICS: Both affixes and clitics are phonologically bound to a neighbouring element, but affixes combine with hosts belonging to just one grammatical category while clitics may occur with different types of host categories. Effectively affixes mark individual words, while clitics mark whole phrases or even whole clauses. 

Many features in Grambank concern both affixes and clitics. The features on prepositions and postpositions are special in this regard: proclitics count as prepositions and enclitics count as postpositions. Since affixes generally mark individual words and not whole phrases, they do not count as adpositions. 

Grambank counts the separate elements of circumfixes as affixes. If one of the elements of a circumfix precedes the host it attaches to, we treat it as a prefix. If one of the elements of a circumfix follows the host it attaches to, we treat it as a suffix.

Grambank does not count nonconcatenative morphology such as ablaut and vocalic melodies or root-and-pattern morphology as affixation.

PARTICLES: In Grambank a particle is a phonologically free grammatical marker that does not inflect.

AGREEMENT: When an element A (e.g. an adnominal demonstrative) is marked for a property C inherent to another element B (e.g. plural number of a noun), A shows agreement with B in C. C does not have to be overtly marked on A for this situation to be called agreement. An adnominal adjective can, for example, agree with its head noun in gender, or a verb can agree with one of its arguments in number. Relevant features are GB170, GB171, GB172, GB177, GB184, GB185, GB186, GB198.

INDEXING VERSUS PRONOUNS: Indexing (also sometimes 'indexation') concerns phonologically bound forms on the verb expressing some features of verbal arguments, such as person, number, and gender. Traditionally, the terms 'verbal agreement', 'concord' and also 'head marking' have been used.

For a marker to be considered an index, it does not matter whether it can or cannot co-occur with nominals that have the same role and reference in the clause. In some grammar writing traditions, if a marker on the verb cannot co-occur with an overt coreferential noun phrase in the same clause, the respective marker on the verb is not considered to be an agreement marker, and might be called a pronoun or a pronominal. For the purposes of this questionnaire, this distinction does not matter: any person-number-etc. marking on the verb – no matter whether it underlies some co-occurrence restrictions and no matter how it is called in the grammar – is a case of indexing. Voice markers are not considered indexes according to this definition since they do not express any features of verbal arguments. If you encounter something that looks like an index, but it does not express person, number or gender, code as "?" and flag this in the comment field.

Pronouns, on the other hand, are defined here as person forms that constitute phonologically independent words. If information about the phonological status of some person form cannot be found, orthography can be used as a proxy: person forms written as separate words can be considered pronouns. Again, note that some descriptions also use the term 'pronoun' for phonologically bound person forms, which are considered indexes here.

CORE ARGUMENTS (S, A and P): We define core arguments as those arguments functioning as the single argument of an intransitive verb (=S), as the patient-like argument of a transitive verb (=P), or as the agent-like argument of a transitive verb (=A). Only major argument marking patterns found with a substantial number of verbs are considered here. A and P can be expected to have only one major argument marking pattern, which occurs with prototypical transitive verbs such as 'kill' and 'destroy', among others. S, on the other hand, has several major argument patterns in some languages. These definitions are relevant for the following features: GB070, GB071, GB089, GB090, GB091, GB092, GB093, GB094, GB095, GB096, GB098, GB099, GB401, GB408, GB409, GB410.

NOUN CLASSES AND CLASSIFIERS: Grambank utilizes the following criteria in distinguishing gender/noun classes from classifiers:

1. Is there a system of nominal classification where some markers vary based on the category of the noun?
2. If the relevant markers are used only or primarily with numerals, demonstratives and possessors (or some subset of those), this is not sufficient to count as gender marking. Consider whether these markers meet our criteria for numeral, demonstrative or possessive classifiers.
3. If these elements systematically co-occur with some other element in the noun phrase or with indexing on the predicate, they should be considered noun class/gender markers.
4. If there are nominal classification markers which do not systematically co-occur with other constituents in the noun phrase or with indexing, then consider whether these markers meet our criteria for classifiers.

There are Four types of classifiers in Grambank:

+ GB058: *Possessive Classifiers*: A system of nominal classification in which classifiers appear only within possessive constructions and are not related to other nominal classification systems such as the gender/noun class of the language, numeral classifiers or demonstrative classifiers.
+ GB057: *Numeral Classifiers*: A system of nominal classification in which cardinal numerals occur with markers that vary depending on the modified noun. These markers are not the same as markers of a gender system used also on noun modifiers other than cardinal numerals, demonstratives, possessives or on predicates.
+ GB038: *Demonstrative Classifiers*: A system of nominal classification in which demonstratives occur with markers to classify the noun. Demonstrative classifiers can have the same form as numeral, possessive or verbal classifiers. These markers are not the same as the markers of a gender system used also on noun modifiers other than cardinal numerals, demonstratives, possessives or on predicates.
+ GB116: *Verbal Classifiers*: A system of nominal classification in which verbal classifiers appear on the verb and classify a noun. Incorporated nouns or suppletive verb stems are subtypes of verbal classifiers. Verbal classifiers can be identical or very similar to demonstrative classifier systems. These markers are not the same as the markers of a gender system.

Be careful to distinguish GB057 and GB198. GB057 asks whether there are numeral classifiers. GB198 (Can an adnominal numeral agree with the noun in noun class/gender?) asks whether the numeral agrees with the noun class/gender of the noun. Languages which answer 1 for either of these look similar, in that either type will have a morpheme used with the numeral that varies depending on the noun.

RELATIVE CLAUSES: Suppose we have a clause with an overt nominal (which is not a pronoun). For example, we might have a clause such as *the woman is tall* where we have an overt nominal *the woman*. We will refer to this nominal as the head (*the woman*) and this clause (*the woman is tall*) as the main clause. A dependent clause is a relative clause if it modifies the head. For example, a dependent clause that can modify *the woman* is *who has a green hat*.

The questions on relative clauses mainly concern the placement of the relative clause with respect to the head nominal. The head noun has two places to fill, one in the main clause and one in the relative clause. Some languages simply let the head nominal act in both clauses as a nexus, while some languages require an overt anaphoric pronoun 'copy' of the head nominal, either in the main clause or in the dependent clause.

1. Suppose now that a language requires no overt pronoun copy in the main clause (i.e. no pronoun copy is required at all, or one is required in the relative clause), the following may happen:
  + GB327: Can the relative clause follow the noun? The entire relative clause comes (immediately) after the head nominal as in *the woman **who has a green hat** is tall*.
  + GB328: Can the relative clause precede the noun? The entire relative clause comes (immediately) before the head nominal as in **who has a green hat** *the woman is tall*.
  + GB329: Are there internally-headed relative clauses? The head nominal occurs inside the relative clause as in **A green hat the woman has** *is tall*.
  + GB331: Are there non-adjacent relative clauses? The last option is that the relative clause is not adjacent to the head nominal as in *the woman is tall* **who has a green hat**, then it is non-adjacent.
+ 2. Suppose now that a language requires a overt 'copy' pronoun in the main clause:
  + GB330: Are there correlative relative clauses? For example the pattern **The woman who has a green hat** *she is tall* counts (only) as correlative (GB330), no matter where in the relative clause the erstwhile (now replaced by an anaphoric pronoun) head nominal occurs.

DEDICATED MARKING: For many of the features it matters whether or not a marker is dedicated to a certain function or whether it also denotes other functions. Exactly what additional functions are allowed differs per feature. So if the marker marks X as well as a different function Y, it counts as a marker dedicated to X (as well as to Y). But if a marker marks something wider than X that often implies X or needs something extra, then it would not be dedicated. Examples:
+ GB037: Do demonstratives show a visible-nonvisible distinction? It cannot be the case that this distinction also marks 'far away' or 'very far away'. Items that are very far away are almost always invisible. If invisibility is only implied by the fact that something is very far away, the language does not have a visible/non-visible distinction per se. This meaning is reached by implication, not by direct denotation. To be certain that the distinction is visible-non visible, it should also be possible to use a 'non-visible' demonstrative if the referent is close by but invisible.
+ GB152: Is there a morphologically marked distinction between simultaneous and sequential clauses? This marker cannot also be the marker for aspect. Many events that are encoded with imperfective aspects might have a secondary implication of simultaneity as events happening in following clauses are clearly happening at the same time. However, this is not enough for a yes in GB152, there needs to be a marker dedicated to simultaneity and/or sequentiality - aspect in this case is to vague and broad a function despite possibly implying simultaneity/sequentiality.
+ GB126: Is there an existential verb? We are looking for an existential and/or locative verb and it cannot be the same as the verb used in property attribution (*you **are** fierce*) or the equative copula (*you **are** a lawyer*). It can be the case that the locative and/or existential verb is also used in predicative possession or comparative constructions, that is allowed. So, certain functions are explicitly allowed and others are explicitly not allowed.

OBLIGATORY MARKING: Many of the features in Grambank specify in the feature description or assume that the marker needs to be obligatory. By grammatical marking, it is generally assumed that always, or nearly always, when the relevant semantics is referred to, the marker is present. Exceptions are possible: there may be suppletive forms for example or special circumstances where the marker/marking is not present. However, the marker/marking should be the most dominant way of expressing the given concept and the concept should nearly always be present when the concept is being expressed.

For example, dual number marking is different from the use of the numeral 'two' in that dual number marking should almost always be present when there is indeed two of something, whereas the numeral 'two' can be left out and under-specified: for example if plural marking is present and the fact that there is two of something isn't important enough.

Obligatoriness is one of the three key concepts of grammar in Grambank. Grammatical markers/marking should be:
+ near-universally obligatory
+ productive (i.e. not restricted to only a subset of lexemes)
+ dedicated.

PRODUCTIVITY: Most features of Grambank are concerned with productive phenomena. A productive rule or pattern is one that can be applied to new words and that is not limited to a small, lexically determined set of words. For example, we want to exclude affixes that only occur in a few fossilized constructions. Since we usually cannot test whether a rule applies to newly invented words, we have to rely on (1) what authors say about the rule, and (2) the perceived frequency of an item to determine its productivity. If an author says that an affix is no longer productive, then it should not be considered for our purposes. If an author says that in a lexicon of 1,000 nouns, the plural marker is applied to only 10 nouns, then we can assume that the plural marker is not productive in this language.

Sometimes, authors do not specify whether a process is productive or not. They may only give one or a few examples, or no examples at all. In this case, **we assume that the process is productive** unless we have evidence of the contrary. Sometimes, productivity is not relevant to a feature. For example, if there are only three auxiliaries in the language and all of them are reduplicated to mark future tense, then we do consider this a relevant morphological rule for Grambank. However, it is logically impossible that this rule applies to a new auxiliary, since the three auxiliaries form a closed class.

GENDER: Grambank uses the term *gender* to describe a grammatical system for categorizing nouns (alternatively called noun class) and not for any social, cultural, biological, or other non-linguistic concepts.

REFERENCING: Sources used in Grambank should be formatted as follows: `Author and Author and Author YEAR:page(s)`. For example: `Hovdhaugen and Mosel 1992:34`. It can happen that your coding is not based on a page or a page range, but on a document as a whole. This may be the case because you did not find evidence of a certain phenomenon throughout a book. In those cases, use the phrase 'in passim' instead of page numbers. For example: `Lichtenberk 1979:in passim`. If you are citing someone who is in their turn getting the information from somewhere else, cite the source you are reading and put the other information in the comment. For example, if Chan (202023) has information on the numeral system of a certain language and this comes from Mead (2020) you are to put `Chan 2023` as the source and `Mead 2020 via Chan 2023` in the comment field.

SOURCE DOCUMENT: