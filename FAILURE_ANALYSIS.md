# Failure analysis — Phase 1 base models (pre-SFT)

Archived outputs only, no inference. Condition **A** uses the 1024-token
re-run; the 512-cap versions were cap-suppressed and are superseded.
Condition **B** (thinking) is unaffected.

Dev split, 1037 (document, hypothesis) pairs per cell, 8 cells.


## 1. Exception / carve-out language — the ContractNLI hypothesis

The ContractNLI paper names negation-by-exception as the dominant failure
of pretrained models. Subset = gold span, or +/-300 chars around it,
containing any of: `except`, `unless`, `provided that`, `other than`, `save for`, `subject to`.

**The comparison is the finding**: error rate on the carve-out subset vs the
non-carve-out subset. Both are restricted to rows that HAVE gold evidence, so
they are like-for-like; the all-rows figure is shown only for context and is
not comparable (it includes NotMentioned rows, which have no gold span).

`STRICT` drops `subject to`, which appears in most commercial contracts
regardless of whether anything is being carved out, and so dilutes the test.

| model | cond | all rows | gold-bearing | carve-out | non-carve | lift | STRICT carve | strict lift |
|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | A | 24.7% | 21.0% | **21.4%** (n=412) | 20.3% (n=202) | 1.05x | **22.2%** (n=378) | 1.09x |
| Qwen3-14B | B | 19.8% | 15.6% | **15.3%** (n=412) | 16.3% (n=202) | 0.94x | **15.1%** (n=378) | 0.92x |
| Qwen3-4B | A | 33.1% | 27.2% | **27.7%** (n=412) | 26.2% (n=202) | 1.05x | **28.8%** (n=378) | 1.10x |
| Qwen3-4B | B | 23.0% | 22.1% | **21.8%** (n=412) | 22.8% (n=202) | 0.96x | **22.5%** (n=378) | 0.99x |
| Qwen3-8B | A | 37.7% | 16.4% | **17.5%** (n=412) | 14.4% (n=202) | 1.22x | **17.7%** (n=378) | 1.23x |
| Qwen3-8B | B | 24.3% | 20.4% | **21.1%** (n=412) | 18.8% (n=202) | 1.12x | **21.4%** (n=378) | 1.14x |
| gemma-4-12B-it | A | 20.8% | 14.3% | **13.1%** (n=412) | 16.8% (n=202) | 0.78x | **14.0%** (n=378) | 0.83x |
| granite-4.2-8b | A | 32.5% | 23.3% | **23.3%** (n=412) | 23.3% (n=202) | 1.00x | **22.8%** (n=378) | 0.98x |

## 2. Evidence false-negative overlap — near-miss or wrong text?

For every evidence FN, token-set overlap of prediction against gold
(|pred ∩ gold| / |gold|). Settles whether models nearly find the right
text or retrieve something else entirely.

| model | cond | n FN | p10 | p50 | p90 | % >0.9 | % >0.5 | % =0 (empty pred) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | A | 434 | 0.00 | 0.55 | 0.97 | 24% | 55% | 16% |
| Qwen3-14B | B | 450 | 0.00 | 0.54 | 0.97 | 18% | 54% | 15% |
| Qwen3-4B | A | 443 | 0.00 | 0.55 | 0.97 | 25% | 54% | 17% |
| Qwen3-4B | B | 460 | 0.11 | 0.63 | 0.97 | 26% | 61% | 9% |
| Qwen3-8B | A | 457 | 0.22 | 0.65 | 0.97 | 28% | 63% | 2% |
| Qwen3-8B | B | 428 | 0.00 | 0.57 | 0.95 | 17% | 55% | 16% |
| gemma-4-12B-it | A | 410 | 0.13 | 0.64 | 0.97 | 22% | 64% | 9% |
| granite-4.2-8b | A | 327 | 0.00 | 0.61 | 0.98 | 26% | 58% | 25% |

## 3. Which policies generate the false conflicts?

Contradiction false positives concentrated by hypothesis, summed across
all cells. If a handful of policies drive the noise, the fix is targeted.

Total Contradiction false positives across all cells: **879**

| hypothesis | false flags | % of all FPs | rate per cell-row | policy |
|---|---|---|---|---|
| nda-7 | 156 | 18% | 32% | Receiving Party may share some Confidential Information with some third-parties (including … |
| nda-15 | 137 | 16% | 28% | Agreement shall not grant Receiving Party any right to Confidential Information. |
| nda-20 | 118 | 13% | 24% | Receiving Party may retain some Confidential Information even after the return or destruct … |
| nda-2 | 100 | 11% | 20% | Confidential Information shall only include technical information. |
| nda-1 | 78 | 9% | 16% | All Confidential Information shall be expressly identified by the Disclosing Party. |
| nda-5 | 62 | 7% | 13% | Receiving Party may share some Confidential Information with some of Receiving Party's emp … |
| nda-10 | 53 | 6% | 11% | Receiving Party shall not disclose the fact that Agreement was agreed or negotiated. |
| nda-13 | 37 | 4% | 8% | Receiving Party may acquire information similar to Confidential Information from a third p … |

Top 8 policies account for **84%** of all false conflicts.


## 4. Contradiction false positives (the commercial bottleneck)

Gold is Entailment or NotMentioned; the model flagged a conflict. Every one of these is a clause escalated to a lawyer for nothing.


### Qwen3-14B (A) — 116 cases, showing 20

**doc 79 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Dealer agrees and acknowledges that the Confidential Information is the property of CIF, and upon termination of this Agreement all rights to the use and ownership of the Confidential Information shall be retained exclusively by CIF. In the event that CIF requests that Dealer return any of its Confidential Information, …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Dealer agrees and acknowledges that the Confidential Information is the property of CIF, and upon termination of this Agreement all rights to the use and ownership of the Confidential Information shall be retained exc …

**doc 435 · nda-10** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* Each of the parties agrees that it will not make use of, disseminate, or in any way disclose any Confidential Information of the other party to any person, firm or business, except to the extent necessary for negotiations, discussions and consultations with personnel or authorized representatives of the other party and …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Each of the parties agrees that it will not make use of, disseminate, or in any way disclose any Confidential Information of the other party to any person, firm or business, except to the extent necessary for negotiat …

**doc 70 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my possession. I understand that my confidentiality obligations under this Agreement will continue after the …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my  …

**doc 82 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* In consideration of our Agreement to supply to you certain written or oral information relating to the Business and entering into discussions relating to the sale of the Business to you (howsoever structured), you the Recipient hereby undertake to us, the Vendor for our benefit and that of the Business, except to the e …  
*Predicted evidence:* 1.1 shall keep secret and confidential our discussions and any information about the Business supplied to you or your advisers by us or by our advisers, whether before or after this Agreement and any analyses or other documents which may be made or prepared by you which contain or otherwise reflect such information or  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1.1 shall keep secret and confidential our discussions and any information about the Business supplied to you or your advisers by us or by our advisers, whether before or after this Agreement and any analyses or oth …

**doc 44 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Attendee will preserve as confidential all Confidential Information that Attendee may obtain or prepare in relation to the Test during the Testing Period. Without BOMI’s prior written consent, which may be given or withheld by BOMI in its sole and absolute discretion, Attendee will not at any time, whether during or af …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Attendee will preserve as confidential all Confidential Information that Attendee may obtain or prepare in relation to the Test during the Testing Period. Without BOMI’s prior written consent, which may be given or wi …

**doc 13 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* 9. “Reviewing Representative” means an employee of a Party who has signed or electronically signed a Non-Disclosure Certificate and who is a principal, partner, officer, director, employee, agent and other representative of a Party.  
*Predicted evidence:* 5. Non-Disclosure to Third Parties. Parties shall not disclose Confidential Information to a third party without the prior written approval of the Disclosing Party. Each Party shall treat all Confidential Information in every form as confidential, and shall not reveal, divulge or disclose Confidential Information, at a …  
*Model said:* {"verdict": "Contradiction", "evidence": ["5. Non-Disclosure to Third Parties. Parties shall not disclose Confidential Information to a third party without the prior written approval of the Disclosing Party. Each Party shall treat all Confidential Information  …

**doc 33 · nda-10** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Gold evidence:* Neither party shall disclose or announce the Purpose to another person without the prior written consent of the other party (such consent not to be unreasonably withheld or delayed) except to the extent that such disclosure or announcement is required as specified in Clause 9.2  
*Predicted evidence:* 9.1 Neither party shall disclose or announce the Purpose to another person without the prior written consent of the other party (such consent not to be unreasonably withheld or delayed) except to the extent that such disclosure or announcement is required as specified in Clause 9.2  
*Model said:* { "verdict": "Contradiction", "evidence": [ "9.1 Neither party shall disclose or announce the Purpose to another person without the prior written consent of the other party (such consent not to be unreasonably withheld or delayed) except to the extent that suc …

**doc 486 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* c) The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and know-how, and will disclose such information only pursuant to the express written permission of the pa …  
*Model said:* {"verdict": "Contradiction", "evidence": ["c) The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and …

**doc 478 · nda-8** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* 6. The Receiving Party may disclose the Disclosing Party's Proprietary Information to the extent required to be disclosed pursuant to final court order; provided, however, that the Receiving Party:  
*Predicted evidence:* The Receiving Party may disclose the Disclosing Party's Proprietary Information to the extent required to be disclosed pursuant to final court order; provided, however, that the Receiving Party: (i) immediately notifies the Disclosing Party upon its receipt of any court order or other document that requests or demands  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The Receiving Party may disclose the Disclosing Party's Proprietary Information to the extent required to be disclosed pursuant to final court order; provided, however, that the Receiving Party: (i) immediately noti …

**doc 44 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* 1. “Confidential Information” shall mean all information, whether communicated in oral, written, electronic or other form, furnished by BOMI or prepared by Attendee during the Testing Period, including, but not limited to, answers and notes made by Attendee during the Testing Period, any test questions and answers, any …  
*Model said:* {"verdict": "Contradiction", "evidence": ["1. “Confidential Information” shall mean all information, whether communicated in oral, written, electronic or other form, furnished by BOMI or prepared by Attendee during the Testing Period, including, but not limite …

**doc 71 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Prospect’s and Prospect’s Broker’s dissemination of any information and materials provided by Listing Broker will be limited to attorneys, accountants, banking representatives, and business advisors directly involved with the above-referenced Property.  
*Predicted evidence:* Prospect’s and Prospect’s Broker’s dissemination of any information and materials provided by Listing Broker will be limited to attorneys, accountants, banking representatives, and business advisors directly involved with the above-referenced Property.  
*Model said:* {"verdict": "Contradiction", "evidence": ["Prospect’s and Prospect’s Broker’s dissemination of any information and materials provided by Listing Broker will be limited to attorneys, accountants, banking representatives, and business advisors directly involved  …

**doc 79 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* The term "Confidential Information" shall mean and include that technical and non-technical information disclosed by CIF to Dealer relating to: (a) CIF's website, information system, and related software and discs, and any data and information contained therein, (b) information and specifications on CIF's products, and …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The term \"Confidential Information\" shall mean and include that technical and non-technical information disclosed by CIF to Dealer relating to: (a) CIF's website, information system, and related software and discs …

**doc 39 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and operation of said project shall be destroyed and files deleted from all of the Bidder’s web sites and ele …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and op …

**doc 478 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 3. The Receiving Party shall preserve Proprietary Information received from the Disclosing Party in confidence by exercising at least the same degree of care used to restrict disclosure and use of the Receiving Party's own similar confidential information, but at least a reasonable degree of care, and shall refrain fro …  
*Predicted evidence:* The Receiving Party shall preserve Proprietary Information received from the Disclosing Party in confidence by exercising at least the same degree of care used to restrict disclosure and use of the Receiving Party's own similar confidential information, but at least a reasonable degree of care, and shall refrain from d …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The Receiving Party shall preserve Proprietary Information received from the Disclosing Party in confidence by exercising at least the same degree of care used to restrict disclosure and use of the Receiving Party's …

**doc 406 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* All Proprietary Information furnished by Health Card to Ciufo is considered loaned for use solely in connection with the Project, and shall be returned by Ciufo to Health Card upon request by Health Card. Ciufo shall certify that it has destroyed or returned all copies of the Proprietary Information in its possession.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "All Proprietary Information furnished by Health Card to Ciufo is considered loaned for use solely in connection with the Project, and shall be returned by Ciufo to Health Card upon request by Health Card. Ciufo shal …

**doc 575 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* 2. Purpose. Vapotherm agrees to disclose the Confidential Information only for the following specific purpose (‘Permitted Purpose’): ________________________________________________________________________________________________ __________________________________________________________________________________________ …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2. Purpose. Vapotherm agrees to disclose the Confidential Information only for the following specific purpose (‘Permitted Purpose’): __________________________________________________________________________________ …

**doc 9 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* VENDOR acknowledges that in the course of providing services to BPS and its students, only VENDOR and its employees or representatives will be given or have access to certain directory, routing, and confidential student information (“Confidential Information”) which may typically protected from disclosure with or witho …  
*Predicted evidence:* Confidential Information. VENDOR acknowledges that in the course of providing services to BPS and its students, only VENDOR and its employees or representatives will be given or have access to certain directory, routing, and confidential student information (“Confidential Information”) which may typically protected fro …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Confidential Information. VENDOR acknowledges that in the course of providing services to BPS and its students, only VENDOR and its employees or representatives will be given or have access to certain directory, routi …

**doc 29 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed in non-tangible form that is identified as confidential at the time of disclosure and summarized in a writing l …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed in non …

**doc 486 · nda-10** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Gold evidence:* d) This Agreement and each additional agreement concluded or written or verbal disclosure made between the Parties, shall be kept confidential and is not to be reproduced, communicated or distributed in any manner whatsoever except on a "need to know" basis to persons directly involved with the closing of any transacti …  
*Predicted evidence:* c) The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and know-how, and will disclose such information only pursuant to the express written permission of the pa …  
*Model said:* {"verdict": "Contradiction", "evidence": ["c) The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and …

**doc 19 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RFP and/or the Permitted Use, which comes into the possession or knowledge of the Confidant, including …  
*Model said:* {"verdict": "Contradiction", "evidence": ["c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the …


### Qwen3-14B (B) — 53 cases, showing 20

**doc 39 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Model said:* <think> Okay, let's see. The user is asking whether the receiving party (Bidder) can create a copy of some Confidential Information under certain circumstances. I need to check the contract to find if there's any mention of copying Confidential Information. Lo …

**doc 16 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* 3. Recipient shall not disclose Confidential Information to any person or entity, except its employees or partners involved in the Project, or to use the Confidential Information for any purpose other than fulfilling its obligations under the Project.  
*Predicted evidence:* Recipient shall not disclose Confidential Information to any person or entity, except its employees or partners involved in the Project, or to use the Confidential Information for any purpose other than fulfilling its obligations under the Project.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract allows the Receiving Party (Recipient) to share Confidential Information with third parties like consultants, agents, or professional advisors. Looking at the contract, specifically section 3. It …

**doc 81 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Either party may, at any time: (a) cease giving Confidential Information to the other party without any liability, and/or (b) request in writing the return or destruction of all or part of its Confidential Information previously disclosed, and all copies thereof, and the receiving party will promptly comply with such r …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party can retain some Confidential Information even after being asked to return or destroy it. First, I need to look at the contract provided. The key section here is probably S …

**doc 15 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* For the purposes of this Agreement the term confidential information includes proprietary materials and information relating thereto including without limitation specifications, drawings, designs, computer software and knowhow.  
*Predicted evidence:* For the purposes of this Agreement the term confidential information includes proprietary materials and information relating thereto including without limitation specifications, drawings, designs, computer software and knowhow.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract states that Confidential Information only includes technical information. The question is about the scope of what's considered confidential. Looking at the contract, the first part under "THE PUR …

**doc 69 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* (b) Promptly following the request of the Disclosing Party, the Receiving Party shall, as directed by the Disclosing Party, (i) destroy (and confirm such destruction in writing to the Disclosing Party) or (ii) return all Information it receives or obtains from a Disclosing Party and all copies thereof, as well as all o …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may retain some Confidential Information even after returning or destroying it, based on the provided contract. First, I need to look at the relevant sections of the contr …

**doc 32 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* The receiving party agrees to immediately return and to deliver all originals and copies of any Confidential Information in its possession or subject to its control to the disclosing party at the address first written above upon the first to occur of the following: termination or expiration of this Agreement; or comple …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party can retain some confidential information even after returning or destroying it, based on the provided contract. First, I need to look through the contract terms. The key s …

**doc 575 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* Receiving Party shall disclose Confidential Information only to those of its employees, agents and independent contractors who have a need to know such information for the Permitted Purpose.  
*Predicted evidence:* 3. Nondisclosure. The Receiving Party agrees that, in consideration of access to the Confidential Information, it shall hold such Confidential Information in strict confidence and shall take all measures necessary to prevent the Confidential Information from falling into the public domain or into the possession of the  …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party can share Confidential Information with third parties like consultants, agents, or advisors. I need to check the contract to find relevant clauses. Looking at the contract, section 3.b say …

**doc 72 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 11. Upon receipt of a written request to the undersigned, the undersigned and its Representatives shall promptly return to BDO any and all Information received, printed, downloaded or otherwise copied together with personal notes including written materials and any notes of verbal conversations with representatives fro …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (the party receiving confidential information under the NDA) may retain some Confidential Information even after returning or destroying it. The contract in question is a  …

**doc 64 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* B. “Delivering Party” has furnished, is furnishing, or will be furnishing, either directly or indirectly through a representative, to the other party (“Receiving Party”) or to the Receiving Party’s representatives, including without limitation, any agents, brokers, lender, appraisers, escrow, professional advisors, con …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Confidential Information shall only include technical information." I need to check the contract to see if that's the case. First, I'll look through t …

**doc 17 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* b. Any VA information, software, applications, computer systems, and hardware accessed by the contractor in the performance of the contract remain the sole property of VA.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Confidential Information shall only include technical information." I need to check the contract text to see if it defines Confidential Information in …

**doc 39 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and operation of said project shall be destroyed and files deleted from all of the Bidder’s web sites and ele …  
*Predicted evidence:* Confidential Information means all information relating to the design, layout and operation of said Project.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract states that Confidential Information only includes technical information. I need to check the contract's definition of Confidential Information. Looking at section 4.B of the contract, it says: " …

**doc 82 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 1.6 shall, immediately upon written request from us, or without such request if you cease to be interested in purchasing the Business, return to us all written Confidential Information and expunge any Confidential Information from any computer, word processor or other similar device in your possession or under your cus …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party (Recipient) can make a copy of Confidential Information under certain circumstances. I need to check the contract provided. Looking at the contract, specifically section 1.6: It says the R …

**doc 14 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 1. Stockholder agrees that the Confidential Material is being produced to them only for purposes reasonably related to Stockholder’s interests as a stockholder of the Company and that Stockholder shall not utilize Confidential Material, any other information produced by the Company or anything derived from Confidential …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that the Agreement does not grant the Receiving Party any right to Confidential Information. First, I need to check the contract terms. The contract is a Confidentiality and …

**doc 39 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Predicted evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract allows the Receiving Party (which is the Bidder here) to share some Confidential Information with third parties like consultants, agents, and professional advisors. Looking at the contract, speci …

**doc 507 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* In consideration of Big Sky's disclosure to Holdings of the Information, Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectiv …  
*Predicted evidence:* Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectively, "Representatives"), in any manner whatsoever, in whole or in part, a …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party (which would be Holdings in this case) can share some Confidential Information with some of their employees. Looking at the contract, the main non-disclosure provisions are in the second p …

**doc 582 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* You hereby agree that the Confidential Information will be used solely for the purpose of evaluating and implementing the Transaction and that such information will be kept confidential by you; provided that any such information may be disclosed to your officers, directors, employees, affiliates, accountants, attorneys …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Confidential Information in the contract is limited to technical information. The contract provided is a Confidentiality Agreement between Verizon and Terremark. First, I need to look thr …

**doc 7 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 3. RETURN OF CONFIDENTIAL INFORMATION Each party shall: (a) within one (1) month of completion of the purpose or receipt of a written request from the other party, return to the other party all documents and materials (and all copies thereof) containing the other party's Confidential Information and certify in writing  …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (which would be the Company in this contract) may retain some Confidential Information even after returning or destroying it. First, I need to look through the contract to …

**doc 405 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* The term "Evaluation Materials" includes information furnished to you orally or in writing (whatever the form or storage medium) or gathered by inspection, and regardless of whether such information is specifically identified as "confidential".  
*Predicted evidence:* The term "Evaluation Materials" includes information furnished to you orally or in writing (whatever the form or storage medium) or gathered by inspection, and regardless of whether such information is specifically identified as "confidential".  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "All Confidential Information shall be expressly identified by the Disclosing Party." First, I need to look through the contract to find any mention of …

**doc 45 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 5. At the instruction of disclosing party, the receiving party agrees to either return or dispose of all information received under this agreement, discontinue any use of the “Confidential Information”, and provide written confirmation thereof.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party can retain some confidential information even after returning or destroying it, based on the provided NDA. First, I need to look through the contract clauses. The key sect …

**doc 39 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and operation of said project shall be destroyed and files deleted from all of the Bidder’s web sites and ele …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party (Bidder) may retain some Confidential Information even after the return or destruction of Confidential Information. First, I need to look through the contract to find rele …


### Qwen3-4B (A) — 114 cases, showing 20

**doc 590 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* 1.2 “Representatives” means as to any person, its directors, officers, employees, agents and advisors (including, without limitation, financial advisors, financing sources, attorneys, accountants and their respective Representatives).  
*Predicted evidence:* 2.1 Non-Disclosure. Without the written consent of Disclosing Party and except as otherwise required by applicable law, Receiving Party shall keep, and shall cause its Representatives to keep, all Proprietary Information confidential and shall not disclose or reveal, and shall cause its Representatives not to disclose  …  
*Model said:* {"verdict": "Contradiction", "evidence": ["2.1 Non-Disclosure. Without the written consent of Disclosing Party and except as otherwise required by applicable law, Receiving Party shall keep, and shall cause its Representatives to keep, all Proprietary Informat …

**doc 605 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made available to Receiving Party, except, in each case, to the extent that Receiving Party is requested pursu …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made av …

**doc 205 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* 3.1. As between the Disclosing Party and the Receiving Party, Confidential Information and all applicable intellectual property rights embodied in the Confidential Information shall remain the sole property of the Disclosing Party.  
*Predicted evidence:* 3.1. As between the Disclosing Party and the Receiving Party, Confidential Information and all applicable intellectual property rights embodied in the Confidential Information shall remain the sole property of the Disclosing Party. Nothing in this Agreement shall be construed as granting any rights to the Receiving Par …  
*Model said:* {"verdict": "Contradiction", "evidence": ["3.1. As between the Disclosing Party and the Receiving Party, Confidential Information and all applicable intellectual property rights embodied in the Confidential Information shall remain the sole property of the Dis …

**doc 547 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 2. Use Restriction. The Receiving Party agrees to use the Confidential Information for the sole purpose of evaluating and negotiating the Business Transaction.  
*Model said:* {"verdict": "Contradiction", "evidence": ["2. Use Restriction. The Receiving Party agrees to use the Confidential Information for the sole purpose of evaluating and negotiating the Business Transaction."]}

**doc 478 · nda-8** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* 6. The Receiving Party may disclose the Disclosing Party's Proprietary Information to the extent required to be disclosed pursuant to final court order; provided, however, that the Receiving Party:  
*Predicted evidence:* 6. The Receiving Party may disclose the Disclosing Party's Proprietary Information to the extent required to be disclosed pursuant to final court order; provided, however, that the Receiving Party: (i) immediately notifies the Disclosing Party upon its receipt of any court order or other document that requests or deman …  
*Model said:* {"verdict": "Contradiction", "evidence": ["6. The Receiving Party may disclose the Disclosing Party's Proprietary Information to the extent required to be disclosed pursuant to final court order; provided, however, that the Receiving Party: (i) immediately not …

**doc 45 · nda-12** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* 3. Notwithstanding any other provisions of this Agreement, each party acknowledges that “Confidential Information” shall not include any information which:  
*Predicted evidence:* 3. Notwithstanding any other provisions of this Agreement, each party acknowledges that "Confidential Information" shall not include any information which: (d) Is, or becomes, independently developed by the receiving party without breach of this Agreement;  
*Model said:* {"verdict": "Contradiction", "evidence": ["3. Notwithstanding any other provisions of this Agreement, each party acknowledges that \"Confidential Information\" shall not include any information which: (d) Is, or becomes, independently developed by the receivin …

**doc 82 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* In consideration of our Agreement to supply to you certain written or oral information relating to the Business and entering into discussions relating to the sale of the Business to you (howsoever structured), you the Recipient hereby undertake to us, the Vendor for our benefit and that of the Business, except to the e …  
*Predicted evidence:* 1.1 shall keep secret and confidential our discussions and any information about the Business supplied to you or your advisers by us or by our advisers, whether before or after this Agreement and any analyses or other documents which may be made or prepared by you which contain or otherwise reflect such information or  …  
*Model said:* {"verdict": "Contradiction", "evidence": ["1.1 shall keep secret and confidential our discussions and any information about the Business supplied to you or your advisers by us or by our advisers, whether before or after this Agreement and any analyses or other …

**doc 507 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* In consideration of Big Sky's disclosure to Holdings of the Information, Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectiv …  
*Predicted evidence:* Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectively, "Representatives"), in any manner whatsoever, in whole or in part, a …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents …

**doc 3 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* “Representatives” shall mean directors, employees, professional advisors or anyone involved with the Party in a professional or business capacity.  
*Predicted evidence:* The Receiving Party undertakes to permit access to the Confidential Information only to its Representatives or employees who require access to such Information solely for the fulfilment of the Permitted Purpose, and furnished on a need-to-know basis and only on the condition that such persons are made aware of the term …  
*Model said:* {"verdict": "Contradiction", "evidence": ["The Receiving Party undertakes to permit access to the Confidential Information only to its Representatives or employees who require access to such Information solely for the fulfilment of the Permitted Purpose, and f …

**doc 13 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* 9. “Reviewing Representative” means an employee of a Party who has signed or electronically signed a Non-Disclosure Certificate and who is a principal, partner, officer, director, employee, agent and other representative of a Party.  
*Predicted evidence:* 5. Non-Disclosure to Third Parties. Parties shall not disclose Confidential Information to a third party without the prior written approval of the Disclosing Party. Each Party shall treat all Confidential Information in every form as confidential, and shall not reveal, divulge or disclose Confidential Information, at a …  
*Model said:* {"verdict": "Contradiction", "evidence": ["5. Non-Disclosure to Third Parties. Parties shall not disclose Confidential Information to a third party without the prior written approval of the Disclosing Party. Each Party shall treat all Confidential Information  …

**doc 79 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* 4. Ownership/Return of Confidential Information.  
*Predicted evidence:* Dealer shall not distribute, disseminate, publish or disclose the Confidential Information to any person, or entity not employed or retained by Dealer or the use of the Confidential Information for any purpose other than to further the marketing and sale of CIF products. Dealer will disclose the Confidential Informatio …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Dealer shall not distribute, disseminate, publish or disclose the Confidential Information to any person, or entity not employed or retained by Dealer or the use of the Confidential Information for any purpose other t …

**doc 10 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Except as set forth in Section 2.06, the Receiving Party shall not disclose any of the Confidential Information or other facts directly related to the Confidential Information to any Person other than its Representatives who have a need to know such Confidential Information and who have been informed of the confidentia …  
*Predicted evidence:* The Receiving Party agrees to use the same degree of care to prevent disclosure of the Confidential Information received by it as the Receiving Party uses to prevent disclosure of its own confidential information, but in no case less than a reasonable degree of care. In addition, except as set forth in section 2.06, wi …  
*Model said:* {"verdict": "Contradiction", "evidence": ["The Receiving Party agrees to use the same degree of care to prevent disclosure of the Confidential Information received by it as the Receiving Party uses to prevent disclosure of its own confidential information, but …

**doc 15 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* 2. The parties will mark or otherwise designate confidential information to show expressly or by necessary implication that it is imparted in confidence.  
*Predicted evidence:* The receiving party will receive all confidential information ... subject to the following conditions: a) it will take all proper and reasonable measures to ensure that the confidentiality of such information is maintained; b) it will not use the information for any commercial purpose or manufacture without obtaining a …  
*Model said:* {"verdict": "Contradiction", "evidence": ["The receiving party will receive all confidential information ... subject to the following conditions: a) it will take all proper and reasonable measures to ensure that the confidentiality of such information is maint …

**doc 69 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* (f) “Representatives” means all Persons acting on behalf of a specified Party or its Affiliates, including without limitation, its directors, officers, lenders, employees, agents, representatives, financial advisors, attorneys, accountants, consultants, reserve and valuation and other experts.  
*Predicted evidence:* Each Receiving Party shall have the right to disclose that it is evaluating the Transaction and transmit Confidential Information to its Representatives only if and to the extent that such Representatives need to know the Confidential Information for the purpose of evaluating the Transaction and are informed by such Re …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Each Receiving Party shall have the right to disclose that it is evaluating the Transaction and transmit Confidential Information to its Representatives only if and to the extent that such Representatives need to know …

**doc 81 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Title or the right to possess Confidential Information as between the parties will remain in the disclosing party.  
*Predicted evidence:* 5. Title. Title or the right to possess Confidential Information as between the parties will remain in the disclosing party.  
*Model said:* {"verdict": "Contradiction", "evidence": ["5. Title. Title or the right to possess Confidential Information as between the parties will remain in the disclosing party."]}

**doc 411 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any business contacts, customers, suppliers, technology …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business o …

**doc 69 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* (b) “ConfidentialInformation” means all information, whether visual, written, electronic or ​ ​ ​  
*Predicted evidence:* The term "Confidential Information" does not include information (i) which is or becomes generally available to the public other than as a result of a disclosure by the Receiving Party or anyone to whom the Receiving Party transmits any Information, (ii) which is or becomes known or available to the Receiving Party on  …  
*Model said:* {"verdict": "Contradiction", "evidence": ["The term \"Confidential Information\" does not include information (i) which is or becomes generally available to the public other than as a result of a disclosure by the Receiving Party or anyone to whom the Receivin …

**doc 79 · nda-3** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information may include verbally conveyed information.  
*Predicted evidence:* Confidential Information shall not include information that: (i) is in or enters the public domain through no improper action or inaction by Dealer, or (ii) was in Dealers possession or known by Dealer prior to receipt from CIF, or (iii) is rightfully disclosed by Dealer by a third party, or (iv) is approved for disclo …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Confidential Information shall not include information that: (i) is in or enters the public domain through no improper action or inaction by Dealer, or (ii) was in Dealers possession or known by Dealer prior to receip …

**doc 73 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Notwithstanding anything to the contrary contained in this Agreement, all of the Confidential Information and the Confidential Materials are and shall remain the property of the Disclosing Party.  
*Predicted evidence:* WHEREAS, the Receiving Party has requested that the Disclosing Party provide to the Receiving Party certain Confidential Information and Confidential Materials (as such terms are defined below) for the Receiving Party to review in order to evaluate a potential business opportunity between the Receiving Party and the Di …  
*Model said:* {"verdict": "Contradiction", "evidence": ["WHEREAS, the Receiving Party has requested that the Disclosing Party provide to the Receiving Party certain Confidential Information and Confidential Materials (as such terms are defined below) for the Receiving Party …

**doc 61 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Company agrees that access to the Confidential Information shall be given by it only to those of its employees who have a need to know to engage in the business relationship contemplated by this Agreement and who have signed a non-use and non-disclosure agreement in content at least as protective of the Confidential In …  
*Predicted evidence:* Company agrees that access to the Confidential Information shall be given by it only to those of its employees who have a need to know to engage in the business relationship contemplated by this Agreement and who have signed a non-use and non-disclosure agreement in content at least as protective of the Confidential In …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Company agrees that access to the Confidential Information shall be given by it only to those of its employees who have a need to know to engage in the business relationship contemplated by this Agreement and who have …


### Qwen3-4B (B) — 80 cases, showing 20

**doc 10 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 2.01 Confidential Information: "Confidential Information" means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party that is clearly marked "Confidential" or clearly intended to be disclosed in confidence.", "2.03 Kept in Co …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the agreement does not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or Not …

**doc 9 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* VENDOR acknowledges that in the course of providing services to BPS and its students, only VENDOR and its employees or representatives will be given or have access to certain directory, routing, and confidential student information (“Confidential Information”) which may typically protected from disclosure with or witho …  
*Predicted evidence:* VENDOR acknowledges that in the course of providing services to BPS and its students, only VENDOR and its employees or representatives will be given or have access to certain directory, routing, and confidential student information ("Confidential Information") which may typically protected from disclosure with or witho …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Confidential Information shall only include technical information." The answer needs to be either Entailment, Contradiction, or NotMentioned, with evi …

**doc 70 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* I understand that access to health information created, received, or maintained by FAU or its affiliates is limited to those who have a valid business, medical, or professional need to know the information.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …

**doc 488 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Neither party acquires any licenses under any intellectual property rights of the other party under this Agreement.  
*Predicted evidence:* 3. Each of the parties agrees that it will not make use of, disseminate, or in any way disclose any Confidential Information of the other party to any person, firm or business, except to the extent necessary for negotiations, discussions, and consultations with personnel or authorized representatives of the other party …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement does not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or Not …

**doc 398 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Seller acknowledges and agrees that these prohibitions against disclosure of Confidential Information are in addition to, and not in lieu of, any rights or remedies that Purchasers or Parent may have available pursuant to the laws of any jurisdiction or at common law to prevent disclosure of trade secrets or proprietar …  
*Predicted evidence:* Seller will keep confidential and will not for a period of three (3) years after the execution of this Agreement, directly or indirectly, divulge to anyone, use or otherwise appropriate any of the Confidential Information for any reason or purpose whatsoever except to authorized representatives of Purchasers or the Par …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement does not grant the Receiving Party (which I assume is the Purchasers or Parent) any right to Confidential Information. First, I need to l …

**doc 482 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 3. The receiving Party shall not, without prior written permission of the disclosing Party, furnish to any third Party any information, equipment or material embodying or made by use of any Confidential Information received or developed hereunder nor use such information for purposes other than internal evaluation so l …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the receiving party to retain some confidential information even after returning or destroying it. The answer needs to be based on the provided contract text. First, I nee …

**doc 15 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* For the purposes of this Agreement the term confidential information includes proprietary materials and information relating thereto including without limitation specifications, drawings, designs, computer software and knowhow.  
*Predicted evidence:* the term confidential information includes proprietary materials and information relating thereto including without limitation specifications, drawings, designs, computer software and knowhow  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Confidential Information shall only include technical information." The answer needs to be either Entailment, Contradiction, or NotMentioned, with evi …

**doc 33 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* No right or licence is granted to either party in relation to the other party's Confidential Information except as expressly set out in this Agreement.  
*Predicted evidence:* 6.1 No right or licence is granted to either party in relation to the other party's Confidential Information except as expressly set out in this Agreement.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …

**doc 39 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 1. Purpose ... for the sole purpose of compiling and submitting a bid in response to the County’s Invitation for Bid to construct the Project.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …

**doc 39 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Predicted evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which is the Bidder in this case) to share some Confidential Information with third parties, including consultants, agents, and professional advisors. …

**doc 28 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* 7. No licenses or rights under any patent, copyright, trademark, or trade secret are granted or are to be implied by this Agreement.  
*Predicted evidence:* 3. The Recipient shall protect such Confidential Information from disclosure to others, using the same degree of care used to protect its own confidential or proprietary information of like importance, but in any case using no less than a reasonable degree of care. The Recipient may disclose Confidential Information re …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …

**doc 610 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* As a condition to such information being furnished to each party, its affiliates, its subsidiaries and other individuals and entities controlled, directly or indirectly, by such party (“Affiliates”), and the respective directors, employees, consultants, accountants, attorneys and advisors of such party and its Affiliat …  
*Predicted evidence:* Neither party will disclose any Confidential Information of the other party to third parties except those of its Associates who are required to have the information in order to evaluate and discuss a Transaction.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to share some Confidential Information with third parties, including consultants, agents, and professional advisors. The answer needs to be either Enta …

**doc 16 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 4. Recipient shall not make copies of the Confidential Information except for those copies required for use by Authorized Employees in the performance of the Agreement. Each copy, including its storage media, shall be marked Confidential, and also include all copyright, trademark and other proprietary notices which app …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (Recipient) to retain some Confidential Information even after returning or destroying it. The answer options are Entailment, Contradiction, or NotMent …

**doc 16 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Recipient may be provided with access to selected information... which helps meet a regulatory purpose or business need...  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the agreement doesn't grant the Recipient any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or NotMention …

**doc 482 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* 6. This Agreement shall not be construed as granting or conferring any rights to license or otherwise, expressly, implied, or otherwise for any invention, discovery or improvement made, conceived or acquired prior to or after the date of this Agreement.  
*Predicted evidence:* 6. This Agreement shall not be construed as granting or conferring any rights to license or otherwise, expressly, implied, or otherwise for any invention, discovery or improvement made, conceived or acquired prior to or after the date of this Agreement.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …

**doc 7 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 3. (b) notwithstanding the completion of the Purpose or return of the documents and materials as aforesaid, continue to be bound by the undertakings set out in Clause 2.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to retain some Confidential Information even after returning or destroying it. The answer needs to be determined based on the contract's terms. First,  …

**doc 17 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 3.b. Any VA information, software, applications, computer systems, and hardware accessed by the contractor in the performance of the contract remain the sole property of VA.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …

**doc 28 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* The Recipient may disclose Confidential Information received hereunder to  
*Predicted evidence:* 3. The Recipient shall protect such Confidential Information from disclosure to others, using the same degree of care... shall not otherwise be disclosed to any third party without the prior written consent of the Owner.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to share some Confidential Information with third parties like consultants, agents, or professional advisors. First, I need to look through the contrac …

**doc 406 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* In consideration of the above promises and the covenants hereinafter set forth, the parties agree as follows: 1. a. "Proprietary Information" means information including, but not limited to clients, pricing and information which is related to the business of Health Card... (Assuming the criteria... Proprietary Informat …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the agreement does not grant the receiving party (Ciufo) any right to confidential information. The answer needs to be either Entailment, Contradiction …

**doc 73 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* The term “Confidential Information” shall not include such portions of the Confidential Information that:  
*Predicted evidence:* Section 5(iv) states: 'shall reveal the Confidential Information and the Confidential Materials only to its Representatives who need to know the Confidential Information and the Confidential Materials in connection with a Possible Transaction, and who (A) are informed by the Receiving Party of the confidential and prop …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may share some Confidential Information with third parties like consultants, agents, and professional advisors, based on the provided contract. First, I need to look throu …


### Qwen3-8B (A) — 204 cases, showing 20

**doc 72 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* The term "Information" does not include and this NDA will not apply to any information that: (a) at the time of disclosure or thereafter is generally available to or known by the public (other than as a result of a disclosure by the undersigned or any of its Representatives in violation of any obligation under this NDA …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The term \"Information\" does not include and this NDA will not apply to any information that: (a) at the time of disclosure or thereafter is generally available to or known by the public (other than as a result of  …

**doc 17 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* The contractor and any subcontractor(s) shall presume that the VA computer systems and storage media that the contractor or subcontractor access have sensitive information and applications, the modification or disclosure of which could cause significant harm or embarrassment to VA beneficiaries and employees and to VA' …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The contractor and any subcontractor(s) shall presume that the VA computer systems and storage media that the contractor or subcontractor access have sensitive information and applications, the modification or discl …

**doc 586 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* the term 'Evaluation Material' means all proprietary and confidential information concerning the Furnishing Party or any of its subsidiaries or affiliates, whether in verbal, visual, written, electronic or other form, which is made available by the Furnishing Party or any of its Representatives to the Receiving Party o …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "the term 'Evaluation Material' means all proprietary and confidential information concerning the Furnishing Party or any of its subsidiaries or affiliates, whether in verbal, visual, written, electronic or other for …

**doc 406 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* b. Proprietary Information does not include any information which  
*Predicted evidence:* Proprietary Information does not include any information which (1) is already known to Ciufo at the time it is disclosed to Ciufo by Health Card, provided that such information has been rightfully received by Ciufo from a third party without restriction on disclosure and without breach of an obligation of confidentiali …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Proprietary Information does not include any information which (1) is already known to Ciufo at the time it is disclosed to Ciufo by Health Card, provided that such information has been rightfully received by Ciufo  …

**doc 582 · nda-18** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall not solicit some of Disclosing Party's representatives.  
*Predicted evidence:* From the date of this Agreement until the earlier to occur of (a) 11:59 p.m. (New York time) on February 3, 2011 and (b) the execution of a definitive agreement between the Company and you relating to a Transaction, except for discussions with you regarding the Transaction, the Company and its affiliates shall not, and …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "From the date of this Agreement until the earlier to occur of (a) 11:59 p.m. (New York time) on February 3, 2011 and (b) the execution of a definitive agreement between the Company and you relating to a Transaction, …

**doc 32 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* “Confidential Information” means confidential and proprietary information (regardless of whether such Confidential Information is in written, oral, visual or electronic form)  
*Predicted evidence:* Confidential Information means confidential and proprietary information (regardless of whether such Confidential Information is in written, oral, visual or electronic form) (a) that consists of Trade Secrets (as hereinafter defined), discoveries, technical, engineering, financial, pricing and/or other business data and …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidential Information means confidential and proprietary information (regardless of whether such Confidential Information is in written, oral, visual or electronic form) (a) that consists of Trade Secrets (as her …

**doc 610 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* The term "Confidential Information" shall be deemed to include those portions of any notes, analyses, compilations, studies, interpretations, memoranda or other documents (regardless of the form thereof) prepared by the receiving party or its Associates which contain, reflect or are based upon, in whole or in part, any …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The term \"Confidential Information\" shall be deemed to include those portions of any notes, analyses, compilations, studies, interpretations, memoranda or other documents (regardless of the form thereof) prepared  …

**doc 605 · nda-8** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Predicted evidence:* Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made available to Receiving Party, except, in each case, to the extent that Receiving Party is requested pursu …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made  …

**doc 39 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Predicted evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s dut …

**doc 482 · nda-8** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Predicted evidence:* 5. In the event of any breach or threatened breach by the receiving Party of its obligations hereunder, the disclosing Party shall have whatever rights and remedies are available to it at law or in equity, provided, however, that in no event shall either Party be liable for any consequential damages of any nature whats …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "5. In the event of any breach or threatened breach by the receiving Party of its obligations hereunder, the disclosing Party shall have whatever rights and remedies are available to it at law or in equity, provided, …

**doc 15 · nda-8** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Predicted evidence:* Under the terms of this Agreement there is no explicit or implied transfer of ownership to the receiving party of any drawings, documents or software, or the copyright subsisting in them. PROVIDED that the obligations herein undertaken will not apply to: ... e) information which is required to be disclosed by legal pro …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Under the terms of this Agreement there is no explicit or implied transfer of ownership to the receiving party of any drawings, documents or software, or the copyright subsisting in them. PROVIDED that the obligatio …

**doc 61 · nda-10** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* Company shall not (i) use any portion of the Confidential Information for any purpose except to evaluate the RFP, including assessment of each Statement of Work under the RFP, in order to prepare its response to the RFP or (ii) disclose the Confidential Information or any part of it to any third party without the prior …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Company shall not (i) use any portion of the Confidential Information for any purpose except to evaluate the RFP, including assessment of each Statement of Work under the RFP, in order to prepare its response to the …

**doc 44 · nda-16** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Gold evidence:* 6. Attendee acknowledges that all  
*Predicted evidence:* Attendee acknowledges that all Test documents, materials, drawings, and other physical media of expression incorporating or containing any Confidential Information that BOMI furnishes to Attendee, or that Attendee prepares during the Testing Period, are BOMI’s property and will be promptly surrendered to BOMI upon the  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Attendee acknowledges that all Test documents, materials, drawings, and other physical media of expression incorporating or containing any Confidential Information that BOMI furnishes to Attendee, or that Attendee p …

**doc 405 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* The term "Evaluation Materials" does not include information which  
*Predicted evidence:* You and your representatives will treat confidentially any information (whether written or oral) that either we or our financial advisor, Credit Suisse First Boston Corporation ("CSFB"), or our other representatives furnish to you in connection with a Possible Transaction involving the Company, together with analyses,  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "You and your representatives will treat confidentially any information (whether written or oral) that either we or our financial advisor, Credit Suisse First Boston Corporation (\"CSFB\"), or our other representativ …

**doc 598 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* Confidential Information includes any reports, analyses or notes that are based on, reflect or contain Confidential Information.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidential Information includes any reports, analyses or notes that are based on, reflect or contain Confidential Information." ] }

**doc 381 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including without limitation, proprietary information of the Company, as it exists from time to time (collectively, "Confidential Information"), are …  
*Predicted evidence:* Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including without limitation, proprietary information of the Company, as it exists from time to time (collectively, "Confidential Information"), are …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including without limitation, proprietary i …

**doc 398 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* Seller acknowledges and agrees that as the owner of the Division it has created substantial confidential information used in and concerning the business and has been afforded an unique opportunity to acquire confidential information concerning the Business and that the misappropriation or disclosure of such confidentia …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Seller acknowledges and agrees that as the owner of the Division it has created substantial confidential information used in and concerning the business and has been afforded an unique opportunity to acquire confide …

**doc 29 · nda-10** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* Recipient will not reproduce, use, disseminate, or disclose CI to any person or entity, except to its employees and authorized representatives (i.e., temporary employees, consultants, and contractors) who need to know the CI to further the Purpose and are bound by confidentiality obligations at least as restrictive as  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Recipient will not reproduce, use, disseminate, or disclose CI to any person or entity, except to its employees and authorized representatives (i.e., temporary employees, consultants, and contractors) who need to kn …

**doc 29 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed in non-tangible form that is identified as confidential at the time of disclosure and summarized in a writing l …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed in non …

**doc 12 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* 1.2 BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not having reasonable need for access to such information, unless CLIENT expressly consents to such disclo …  
*Predicted evidence:* BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not having reasonable need for access to such information, unless CLIENT expressly consents to such disclosure …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not h …


### Qwen3-8B (B) — 75 cases, showing 20

**doc 406 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 9. All Proprietary Information furnished by Health Card to Ciufo is considered loaned for use solely in connection with the Project, and shall be returned by Ciufo to Health Card upon request by Health Card. Ciufo shall certify that it has destroyed or returned all copies of the Proprietary Information in its possessio …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party (which would be Ciufo in this case) may retain some Confidential Information even after returning or destroying it. First, I need t …

**doc 69 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* (b) Promptly following the request of the Disclosing Party, the Receiving Party shall, as directed by the Disclosing Party, (i) destroy (and confirm such destruction in writing to the Disclosing Party) or (ii) return all Information it receives or obtains from a Disclosing Party and all copies thereof, as well as all o …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to retain some Confidential Information even after returning or destroying it. First, I need to look at the relevant sections of the contract. The ques …

**doc 14 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* 2. The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as provided in this Confidentiality Agreement to end; provided that in no event shall stockholder, direct …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which would be the Stockholder here) to share some Confidential Information with third parties like consultants, agents, or professional advisors. Fir …

**doc 15 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* For the purposes of this Agreement the term confidential information includes proprietary materials and information relating thereto including without limitation specifications, drawings, designs, computer software and knowhow.  
*Predicted evidence:* the term confidential information includes proprietary materials and information relating thereto including without limitation specifications, drawings, designs, computer software and knowhow  
*Model said:* <think> Okay, let's tackle this. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that "Confidential Information shall only include technical information." First, I need to look at the contract's definition of con …

**doc 19 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 4. Return of Confidential Information: The Confidant shall immediately on notice at any time from City return to City, or destroy, any and all Confidential Information in accordance with City’s direction.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information." First, I need to understa …

**doc 28 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* 1. "Confidential Information" means information that relates to the purpose stated in Section 2 below that has been designated as proprietary by the disclosing party with an appropriate conspicuous stamp or legend on the document exchanged or, although not related to such purpose, is nevertheless disclosed, and that sh …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that "Confidential Information shall only include technical information." First, I need to look at the contract's definiti …

**doc 13 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 8. Ownership and Use of Confidential Information. All Confidential Information delivered by a Disclosing Party to a Party pursuant to this Agreement shall be and remain the property of the Disclosing Party, and such Confidential Information shall be promptly returned to the Disclosing Party upon request. Once the Party …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information. First, I need to underst …

**doc 15 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* 4. Under the terms of this Agreement there is no explicit or implied transfer of ownership to the receiving party of any drawings, documents or software, or the copyright subsisting in them.  
*Predicted evidence:* 4. Under the terms of this Agreement there is no explicit or implied transfer of ownership to the receiving party of any drawings, documents or software, or the copyright subsisting in them. PROVIDED that the obligations herein undertaken will not apply to...  
*Model said:* <think> Okay, let's tackle this question. The user wants to know if the contract supports, contradicts, or doesn't mention the statement that "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to look through the c …

**doc 595 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* You agree that the Evaluation Material is and shall remain the property of the Company or its subsidiaries and affiliates, as applicable, and that neither the Company nor any of its Representatives, subsidiaries or affiliates has granted you any license, copyright, or similar right with respect to any of the Evaluation …  
*Predicted evidence:* You and your Representatives (i) will use the Evaluation Material solely for the purpose of evaluating a possible Transaction with the Company...  
*Model said:* <think> Okay, let's tackle this query. The user wants to know if the contract supports, contradicts, or doesn't mention the statement: "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to understand the statement. …

**doc 44 · nda-5** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Predicted evidence:* Attendee will not at any time, whether during or after the Testing Period, (a) disclose any Confidential Information to any third party nor give any third party access thereto, nor (b) use any Confidential Information except as required in taking the Test.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which in this case is the Attendee) to share some Confidential Information with some of their employees. First, I need to look at the contract terms.  …

**doc 398 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Seller acknowledges and agrees that these prohibitions against disclosure of Confidential Information are in addition to, and not in lieu of, any rights or remedies that Purchasers or Parent may have available pursuant to the laws of any jurisdiction or at common law to prevent disclosure of trade secrets or proprietar …  
*Predicted evidence:* Seller acknowledges and agrees that as the owner of the Division it has created substantial confidential information used in and concerning the business and has been afforded an unique opportunity to acquire confidential information concerning the Business and that the misappropriation or disclosure of such confidentia …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports, contradicts, or doesn't mention the statement: "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to look through the cont …

**doc 39 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and operation of said project shall be destroyed and files deleted from all of the Bidder’s web sites and ele …  
*Predicted evidence:* Confidential Information means all information relating to the design, layout and operation of said Project.  
*Model said:* <think> Okay, let's tackle this query. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that "Confidential Information shall only include technical information." First, I need to look at the contract provided. The …

**doc 64 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* B. "Delivering Party" has furnished, is furnishing, or will be furnishing, either directly or indirectly through a representative, to the other party ("Receiving Party") or to the Receiving Party’s representatives... certain information, in whatever form or by whatever mode or medium whatsoever, about the Delivering Pa …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that "Confidential Information shall only include technical information." First, I need to look through the contract to fi …

**doc 507 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* In consideration of Big Sky's disclosure to Holdings of the Information, Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectiv …  
*Predicted evidence:* Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectively, "Representatives"), in any manner whatsoever, in whole or in part...  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which would be Holdings in this case) to share some Confidential Information with some of their employees. First, I need to recall the relevant parts  …

**doc 478 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 3. The Receiving Party shall preserve Proprietary Information received from the Disclosing Party in confidence by exercising at least the same degree of care used to restrict disclosure and use of the Receiving Party's own similar confidential information, but at least a reasonable degree of care, and shall refrain fro …  
*Predicted evidence:* 3. The Receiving Party shall preserve Proprietary Information received from the Disclosing Party in confidence by exercising at least the same degree of care used to restrict disclosure and use of the Receiving Party's own similar confidential information, but at least a reasonable degree of care, and shall refrain fro …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may share some Confidential Information with some of their employees, based on the provided NDA. First, I need to look at the relevant sections of the contract. The key pa …

**doc 605 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* From time to time, Blueknight and the General Partner may (but shall not be obligated to) disclose to Receiving Party certain "non-public" information relating to Blueknight, the General Partner, their respective Affiliates and their respective businesses.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Agreement shall not grant Receiving Party any right to Confidential Information." So, I need to check the contract to see if it mentions anything abou …

**doc 37 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* The Receiving Party acknowledges and agrees that all property, including intellectual property rights, in Confidential Information disclosed to it by the Disclosing Party shall remain with and be vested in the Disclosing Party or relevant member of the Disclosing Party Group.  
*Predicted evidence:* 6.1 The Receiving Party acknowledges and agrees that all property, including intellectual property rights, in Confidential Information disclosed to it by the Disclosing Party shall remain with and be vested in the Disclosing Party or relevant member of the Disclosing Party Group.  
*Model said:* <think> Okay, let's tackle this question. The user wants to know if the contract supports the position that "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to look through the contract to find relevant sections. …

**doc 9 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* VENDOR, its employees and/or representatives, shall use the Confidential Information and routing data solely in connection with performance by VENDOR of the services provided to BPS as outline by this agreement and pursuant to the Challenge Document and for no other purpose.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Agreement shall not grant Receiving Party any right to Confidential Information." I need to check the contract to see if it mentions anything about th …

**doc 20 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 3.1 All Confidential Information disclosed by the Disclosing Party to the Receiving Party shall remain the property of the Disclosing Party. The Receiving Party shall not be entitled to copy any Confidential Information furnished by the Disclosing Party hereunder, unless and to the extent it is necessary for the purpos …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may retain some Confidential Information even after returning or destroying it. The contract in question is a Mutual Non-Disclosure Agreement. First, I need to look at the …

**doc 39 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and operation of said project shall be destroyed and files deleted from all of the Bidder’s web sites and ele …  
*Model said:* <think> Okay, let's tackle this query. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that the Receiving Party (Bidder) may retain some Confidential Information even after the return or destruction of Confidenti …


### gemma-4-12B-it (A) — 79 cases, showing 20

**doc 69 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* (ii) destroy (and confirm such destruction in writing to the Disclosing Party) or (iii) return all Information it receives or obtains from a Disclosing Party and all copies thereof, as well as all other materials containing Confidential Information in the Receiving Party’s possession (including, without limitation, not …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["(ii) destroy (and confirm such destruction in writing to the Disclosing Party) or (iii) return all Information it receives or obtains from a Disclosing Party and all copies thereof, as well as all other materi …

**doc 547 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* The Receiving Party will disclose Confidential Information only to its employees and to its representative whose participation is necessary or helpful to evaluate and advance to the Business Transaction and who have agreed to be bound by the terms of this Agreement.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The Receiving Party will disclose Confidential Information only to its employees and to its representative whose participation is necessary or helpful to evaluate and advance to the Business Transaction and wh …

**doc 15 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* c) it will not disclose the information to any third party without written permission;  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["c) it will not disclose the information to any third party without written permission;"]} ```

**doc 486 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and know-how, and will disclose such information only pursuant to the express written permission of the party …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledg …

**doc 19 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RFP and/or the Permitted Use, which comes into the possession or knowledge of the Confidant, including …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the C …

**doc 69 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* The exposure of the Receiving Party to the Confidential Information or its disclosure shall not be construed in any way to grant to the Receiving Party any right or license with respect to the Information other than the right to use such Confidential Information strictly in accordance with this Agreement.  
*Predicted evidence:* The exposure of the Receiving Party to the Confidential Information or its disclosure shall not be construed in any way to grant to the Receiving Party any right or license with respect to the Information other than the right to use such Confidential Information strictly in accordance with this Agreement.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The exposure of the Receiving Party to the Confidential Information or its disclosure shall not be construed in any way to grant to the Receiving Party any right or license with respect to the Information othe …

**doc 507 · nda-16** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Gold evidence:* Big Sky and Holdings agree that, at the conclusion of their review of the Information, or within three business days of the other's request, all copies of the Information, in any form whatsoever, will be delivered by Holdings and its Representatives to Big Sky.  
*Predicted evidence:* Big Sky and Holdings agree that, at the conclusion of their review of the Information, or within three business days of the other's request, all copies of the Information, in any form whatsoever, will be delivered by Holdings and its Representatives to Big Sky.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Big Sky and Holdings agree that, at the conclusion of their review of the Information, or within three business days of the other's request, all copies of the Information, in any form whatsoever, will be deliv …

**doc 35 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 2.4 The Receiving Party acknowledges that the Disclosing Party is under no obligation to disclose any Confidential Information or any other information or documentation relating to the Transaction to the Receiving Party, notwithstanding the conclusion of this Agreement. The Receiving Party further acknowledges that, as …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["2.4 The Receiving Party acknowledges that the Disclosing Party is under no obligation to disclose any Confidential Information or any other information or documentation relating to the Transaction to the Recei …

**doc 411 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any business contacts, customers, suppliers, technology …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other bu …

**doc 13 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* Confidential Information that is in writing or other tangible form (including electronic form) shall be subject to this Agreement only if it is clearly marked as "Confidential" when disclosed by the Disclosing Party to a Party.  
*Predicted evidence:* Confidential Information that is in writing or other tangible form (including electronic form) shall be subject to this Agreement only if it is clearly marked as "Confidential" when disclosed by the Disclosing Party to a Party. Confidential Information that is provided orally shall be subject to this Agreement only if  …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Confidential Information that is in writing or other tangible form (including electronic form) shall be subject to this Agreement only if it is clearly marked as \"Confidential\" when disclosed by the Disclosi …

**doc 44 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* In consideration for your right to take the CBT Test, Attendee agrees to the following:  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["In consideration for your right to take the CBT Test, Attendee agrees to the following:", "Attendee will preserve as confidential all Confidential Information that Attendee may obtain or prepare in relation to …

**doc 82 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* In consideration of our Agreement to supply to you certain written or oral information relating to the Business and entering into discussions relating to the sale of the Business to you (howsoever structured), you the Recipient hereby undertake to us, the Vendor for our benefit and that of the Business, except to the e …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["In consideration of our Agreement to supply to you certain written or oral information relating to the Business and entering into discussions relating to the sale of the Business to you (howsoever structured), …

**doc 39 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and operation of said project shall be destroyed and files deleted from all of the Bidder’s web sites and ele …  
*Predicted evidence:* Confidential Information means all information relating to the design, layout and operation of said Project.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Confidential Information means all information relating to the design, layout and operation of said Project."]} ```

**doc 575 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* All Inventions will be immediately reported to Vapotherm and shall be the exclusive property of Vapotherm.  
*Predicted evidence:* No license or other right is created or granted hereby, except the specific right to receive the Confidential Information and evaluate it as set forth herein, nor shall any license or any other right with respect to the subject matter hereof be created or granted except by written agreement signed by the duly authorize …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["No license or other right is created or granted hereby, except the specific right to receive the Confidential Information and evaluate it as set forth herein, nor shall any license or any other right with resp …

**doc 29 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed in non-tangible form that is identified as confidential at the time of disclosure and summarized in a writing l …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed  …

**doc 81 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “proprietary,” “secret,” or similar legend, and (ii) discussions relating to that information whether those discussio …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “propr …

**doc 39 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder …

**doc 590 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* 1.1 “Proprietary Information” means any and all information and material disclosed by Disclosing Party or any of its Representatives to Receiving Party or any of its Representatives in connection with the Proposed Transaction or in the course of the parties’ evaluation and negotiation of the Proposed Transaction, toget …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["1.1 “Proprietary Information” means any and all information and material disclosed by Disclosing Party or any of its Representatives to Receiving Party or any of its Representatives in connection with the Prop …

**doc 13 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* No license to the Party, under any trademark, patent copyright, mask work protection right or any other intellectual property right, is either granted or implied by the conveying of Confidential Information to such Party.  
*Predicted evidence:* No license to the Party, under any trademark, patent copyright, mask work protection right or any other intellectual property right, is either granted or implied by the conveying of Confidential Information to such Party.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["No license to the Party, under any trademark, patent copyright, mask work protection right or any other intellectual property right, is either granted or implied by the conveying of Confidential Information to …

**doc 14 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* WHEREAS, pursuant to the Demand and subject to the Stockholder executing this Agreement, the Company has agreed to allow the Stockholder to inspect the stockholder list (which, together with information or analyses derived from information made available for inspection, is referred to as the “Confidential Material”);  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["WHEREAS, pursuant to the Demand and subject to the Stockholder executing this Agreement, the Company has agreed to allow the Stockholder to inspect the stockholder list (which, together with information or ana …


### granite-4.2-8b (A) — 158 cases, showing 20

**doc 28 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 4. The restrictions of this Agreement on use and disclosure of Confidential Information shall not apply to information, which is:  
*Predicted evidence:* 4. The restrictions of this Agreement on use and disclosure of Confidential Information shall not apply to information, which is: (a) in the public domain as of the date of this Agreement or which later comes into the domain from a source other than the Recipient; (b) in the Recipient’s possession free of any obligatio …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "4. The restrictions of this Agreement on use and disclosure of Confidential Information shall not apply to information, which is:\n(a) in the public domain as of the date of this Agreement or which later comes into  …

**doc 381 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 2. Protection of Information. Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including without limitation, proprietary information of the Company, as it exists from time to time (collectively, "C …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2. Protection of Information. Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including wit …

**doc 14 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 1. Stockholder agrees that the Confidential Material is being produced to them only for purposes reasonably related to Stockholder’s interests as a stockholder of the Company and that Stockholder shall not utilize Confidential Material, any other information produced by the Company or anything derived from Confidential …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1. Stockholder agrees that the Confidential Material is being produced to them only for purposes reasonably related to Stockholder’s interests as a stockholder of the Company and that Stockholder shall not utilize C …

**doc 81 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “proprietary,” “secret,” or similar legend, and (ii) discussions relating to that information whether those discussio …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “proprietary …

**doc 71 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluation.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluat …

**doc 9 · nda-13** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Predicted evidence:* VENDOR acknowledges that in the course of providing services to BPS and its students, only VENDOR and its employees or representatives will be given or have access to certain directory, routing, and confidential student information (“Confidential Information”) which may typically protected from disclosure with or witho …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "VENDOR acknowledges that in the course of providing services to BPS and its students, only VENDOR and its employees or representatives will be given or have access to certain directory, routing, and confidential stu …

**doc 16 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 7. The obligation of non-disclosure shall not extend to:  
*Predicted evidence:* 7. The obligation of non-disclosure shall not extend to: (1) information which is then already in the possession of the Recipient and not under a duty of non-disclosure; (2) information which is generally known or revealed to the public other than by breach of this Agreement; (3) information which is revealed to the Re …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "7. The obligation of non-disclosure shall not extend to: (1) information which is then already in the possession of the Recipient and not under a duty of non-disclosure; (2) information which is generally known or r …

**doc 13 · nda-2** — gold `Entailment`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* 4. “Confidential Information” means a Party’s energy management system model data provided or to be provided by a Disclosing Party to another Party.  
*Predicted evidence:* 4. ‘Confidential Information’ means a Party’s energy management system model data provided or to be provided by a Disclosing Party to another Party.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "4. ‘Confidential Information’ means a Party’s energy management system model data provided or to be provided by a Disclosing Party to another Party.", "8. All Confidential Information delivered by a Disclosing Party …

**doc 19 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 4. Return of Confidential Information: The Confidant shall immediately on notice at any time from City return to City, or destroy, any and all Confidential Information in accordance with City’s direction.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "4. Return of Confidential Information: The Confidant shall immediately on notice at any time from City return to City, or destroy, any and all Confidential Information in accordance with City’s direction.", "2. Acce …

**doc 33 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* This Agreement does not apply to Confidential Information:  
*Predicted evidence:* 5. EXCEPTIONS This Agreement does not apply to Confidential Information: 5.1 to the extent that the Confidential Information is or comes into the public domain other than by breach of this Agreement; 5.2 which the Receiving Party can show by its written or other records was in its possession prior to receipt from the D …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "5. EXCEPTIONS\nThis Agreement does not apply to Confidential Information:\n5.1 to the extent that the Confidential Information is or comes into the public domain other than by breach of this Agreement;\n5.2 which th …

**doc 9 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* VENDOR, its employees and/or representatives, shall use the Confidential Information and routing data solely in connection with performance by VENDOR of the services provided to BPS as outline by this agreement and pursuant to the Challenge Document and for no other purpose.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "VENDOR, its employees and/or representatives, shall use the Confidential Information and routing data solely in connection with performance by VENDOR of the services provided to BPS as outline by this agreement and  …

**doc 82 · nda-12** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* 1.2 shall not use any Confidential Information for your own purposes or to obtain a commercial, trading or other advantage, but shall use such Confidential Information for the sole purpose of evaluating and negotiating the proposed sale of the Business, and, save as set out above, shall not permit any Confidential Info …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1.2 shall not use any Confidential Information for your own purposes or to obtain a commercial, trading or other advantage, but shall use such Confidential Information for the sole purpose of evaluating and negotiat …

**doc 19 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 2. Access/Use of Confidential Information: Subject to the terms and conditions of this Agreement, Confidant may use the Confidential Information only for the Permitted Use and for no other purpose whatsoever.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2. Access/Use of Confidential Information: Subject to the terms and conditions of this Agreement, Confidant may use the Confidential Information only for the Permitted Use and for no other purpose whatsoever.", "3.  …

**doc 575 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* Receiving Party shall disclose Confidential Information only to those of its employees, agents and independent contractors who have a need to know such information for the Permitted Purpose.  
*Predicted evidence:* 2. Purpose. Vapotherm agrees to disclose the Confidential Information only for the following specific purpose (‘Permitted Purpose’): ________________________________________________________________________________________________ __________________________________________________________________________________________ …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2. Purpose. Vapotherm agrees to disclose the Confidential Information only for the following specific purpose (‘Permitted Purpose’): __________________________________________________________________________________ …

**doc 17 · nda-17** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Gold evidence:* f. The contractor and its employees shall not make any copies of any VA information, including software or applications that are not copyrighted, except as necessary to restore VA computer systems or data storage devices to an operating state.  
*Predicted evidence:* f. The contractor and its employees shall not make any copies of any VA information, including software or applications that are not copyrighted, except as necessary to restore VA computer systems or data storage devices to an operating state.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "f. The contractor and its employees shall not make any copies of any VA information, including software or applications that are not copyrighted, except as necessary to restore VA computer systems or data storage de …

**doc 54 · nda-12** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* 4. The DISCLOSEE also agrees not to use the confidential information other than for the purpose as provided herein and not to use that information commercially in any way without the DISCLOSER's consent in writing.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "4. The DISCLOSEE also agrees not to use the confidential information other than for the purpose as provided herein and not to use that information commercially in any way without the DISCLOSER's consent in writing." …

**doc 19 · nda-13** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Predicted evidence:* Confidant may use the Confidential Information only for the Permitted Use and for no other purpose whatsoever.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidant may use the Confidential Information only for the Permitted Use and for no other purpose whatsoever.", "Confidant agrees that: Confidential Information shall be kept in the strictest confidence without lim …

**doc 14 · nda-12** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* 1. Stockholder agrees that the Confidential Material is being produced to them only for purposes reasonably related to Stockholder’s interests as a stockholder of the Company and that Stockholder shall not utilize Confidential Material, any other information produced by the Company or anything derived from Confidential …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1. Stockholder agrees that the Confidential Material is being produced to them only for purposes reasonably related to Stockholder’s interests as a stockholder of the Company and that Stockholder shall not utilize C …

**doc 13 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* Confidential Information excludes any information that:  
*Predicted evidence:* 4. "Confidential Information" means a Party’s energy management system model data provided or to be provided by a Disclosing Party to another Party. Confidential Information shall be disclosed only to Reviewing Representatives and only used to enable a Party to perform its Transmission Functions.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "4. \"Confidential Information\" means a Party’s energy management system model data provided or to be provided by a Disclosing Party to another Party. Confidential Information shall be disclosed only to Reviewing Re …

**doc 33 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 2.2 not to use the other party's Confidential Information except for the Purpose;  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2.2 not to use the other party's Confidential Information except for the Purpose;", "2.3 not to disclose the other party's Confidential Information to another person and to use all reasonable efforts to prevent any  …


## 5. Failed abstention

Gold is NotMentioned; the model forced a verdict on a clause that does not address the policy at all.


### Qwen3-14B (A) — 127 cases, showing 20

**doc 82 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* 1.1 shall keep secret and confidential our discussions and any information about the Business supplied to you or your advisers by us or by our advisers, whether before or after this Agreement and any analyses or other documents which may be made or prepared by you which contain or otherwise reflect such information or  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1.1 shall keep secret and confidential our discussions and any information about the Business supplied to you or your advisers by us or by our advisers, whether before or after this Agreement and any analyses or oth …

**doc 82 · nda-3** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Predicted evidence:* In consideration of our Agreement to supply to you certain written or oral information relating to the Business and entering into discussions relating to the sale of the Business to you (howsoever structured), you the Recipient hereby undertake to us, the Vendor for our benefit and that of the Business, except to the e …  
*Model said:* { "verdict": "Entailment", "evidence": [ "In consideration of our Agreement to supply to you certain written or oral information relating to the Business and entering into discussions relating to the sale of the Business to you (howsoever structured), you the  …

**doc 205 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* 1.1. "Confidential Information" shall mean information that is revealed by or obtained through a party (whether in writing, orally or by any other means) (a "Disclosing Party") to the other (a "Receiving Party") including, without limitation, (i) all forms and types of financial, business, scientific, technical, econom …  
*Model said:* {"verdict": "Contradiction", "evidence": ["1.1. \"Confidential Information\" shall mean information that is revealed by or obtained through a party (whether in writing, orally or by any other means) (a \"Disclosing Party\") to the other (a \"Receiving Party\") …

**doc 598 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* Confidential Information means all information which is non-public, confidential or proprietary in nature, whether transferred in writing, orally, visually, electronically or by other means, disclosed by the Corporation or its representatives to the Recipient on or after the date hereof and for the Purpose.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidential Information means all information which is non-public, confidential or proprietary in nature, whether transferred in writing, orally, visually, electronically or by other means, disclosed by the Corpora …

**doc 16 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 5. Notes and other documents referencing or relating to Confidential Information may be made and kept by the Recipient, but shall continue to be governed by this Confidentiality Agreement until they are destroyed.  
*Model said:* {"verdict": "Entailment", "evidence": ["5. Notes and other documents referencing or relating to Confidential Information may be made and kept by the Recipient, but shall continue to be governed by this Confidentiality Agreement until they are destroyed."]}

**doc 44 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 6. Attendee acknowledges that all Test documents, materials, drawings, and other physical media of expression incorporating or containing any Confidential Information that BOMI furnishes to Attendee, or that Attendee prepares during the Testing Period, are BOMI’s property and will be promptly surrendered to BOMI upon t …  
*Model said:* {"verdict": "Contradiction", "evidence": ["6. Attendee acknowledges that all Test documents, materials, drawings, and other physical media of expression incorporating or containing any Confidential Information that BOMI furnishes to Attendee, or that Attendee  …

**doc 27 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Recipient shall reproduce Discloser's proprietary rights notices on any copies, in the same manner in which such notices were set forth in or on the original.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Recipient shall reproduce Discloser's proprietary rights notices on any copies, in the same manner in which such notices were set forth in or on the original." ] }

**doc 61 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* 1. Company agrees that all information, documents, data, materials and the AOC proprietary software systems disclosed to, or accessed by, Company in connection with, or related to, the RFP or the RFP process, in any form whether oral or written, or in any medium, including but not limited to any of the following is "Co …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1. Company agrees that all information, documents, data, materials and the AOC proprietary software systems disclosed to, or accessed by, Company in connection with, or related to, the RFP or the RFP process, in any …

**doc 69 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* Promptly following the request of the Disclosing Party, the Receiving Party shall, as directed by the Disclosing Party, (i) destroy (and confirm such destruction in writing to the Disclosing Party) or (ii) return all Information it receives or obtains from a Disclosing Party and all copies thereof, as well as all other …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Promptly following the request of the Disclosing Party, the Receiving Party shall, as directed by the Disclosing Party, (i) destroy (and confirm such destruction in writing to the Disclosing Party) or (ii) return all I …

**doc 14 · nda-5** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Predicted evidence:* 2. The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as provided in this Confidentiality Agreement to end; provided that in no event shall stockholder, direct …  
*Model said:* {"verdict": "Contradiction", "evidence": ["2. The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as …

**doc 81 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “proprietary,” “secret,” or similar legend, and (ii) discussions relating to that information whether those discussio …  
*Model said:* {"verdict": "Contradiction", "evidence": ["The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “proprietary,” …

**doc 586 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Notwithstanding the return, destruction or erasure of Evaluation Material hereunder, the Receiving Party and its Representatives shall continue to be bound by their confidentiality and other obligations hereunder.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Notwithstanding the return, destruction or erasure of Evaluation Material hereunder, the Receiving Party and its Representatives shall continue to be bound by their confidentiality and other obligations hereunder." ] }

**doc 10 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* 2.01 Confidential Information means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party that is clearly marked “Confidential” or clearly intended to be disclosed in confidence.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2.01 Confidential Information means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party that is clearly marked “Confid …

**doc 486 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* c) The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and know-how, and will disclose such information only pursuant to the express written permission of the pa …  
*Model said:* {"verdict": "Contradiction", "evidence": ["c) The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and …

**doc 411 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any ... Confidential Information ... which the Stockhol …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business …

**doc 563 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Notwithstanding the immediately preceding sentence, the parties agree that destruction will not be made of (i) any electronic copy of any Confidential Information that was created pursuant to the standard electronic backup and archival procedures of the Recipient if (x) personnel of the Recipient whose functions are no …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Notwithstanding the immediately preceding sentence, the parties agree that destruction will not be made of (i) any electronic copy of any Confidential Information that was created pursuant to the standard electronic ba …

**doc 15 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* The receiving party will not disclose the information to any third party without written permission;  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The receiving party will not disclose the information to any third party without written permission;" ] }

**doc 70 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my possession. I understand that my confidentiality obligations under this Agreement will continue after the …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my  …

**doc 29 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Recipient will not reproduce, use, disseminate, or disclose CI to any person or entity, except to its employees and authorized representatives (i.e., temporary employees, consultants, and contractors) who need to know the CI to further the Purpose and are bound by confidentiality obligations at least as restrictive as  …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Recipient will not reproduce, use, disseminate, or disclose CI to any person or entity, except to its employees and authorized representatives (i.e., temporary employees, consultants, and contractors) who need to know  …

**doc 14 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as provided in this Confidentiality Agreement to end; provided that in no event shall stockholder, directly  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as  …


### Qwen3-14B (B) — 109 cases, showing 20

**doc 29 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Recipient will not reproduce, use, disseminate, or disclose CI to any person or entity, except to its employees and authorized representatives (i.e., temporary employees, consultants, and contractors) who need to know the CI to further the Purpose and are bound by confidentiality obligations at least as restrictive as  …  
*Model said:* <think> Okay, let's see. The user is asking whether the receiving party (Recipient) can create a copy of some Confidential Information under certain circumstances. I need to check the contract provided. Looking at the NDA, section 2 talks about Confidentiality …

**doc 15 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* 3c) it will not disclose the information to any third party without written permission;  
*Model said:* <think> Okay, let's see. The user is asking whether the receiving party can share confidential information with third parties like consultants, agents, or advisors. I need to check the contract terms. Looking at clause 3c, it says the receiving party can't dis …

**doc 61 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 5. Except as otherwise provided in any future agreement, at any time, Company agrees that it shall return to the AOC upon request all documents, records, notebooks, computer media or other stored information of any form or type whatsoever containing any Confidential Information, including all copies thereof, then in it …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (which would be the Company in this contract) may retain some Confidential Information even after returning or destroying it. First, I need to look through the contract to …

**doc 37 · nda-10** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* 1.1(c) discussions, negotiations, and correspondence between the Disclosing Party or any of its directors, officers, employees, consultants or professional advisers and the Receiving Party or any of its directors, officers, employees, consultants and professional advisers in connection with the Permitted Purpose and al …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party is prohibited from disclosing the fact that the Agreement was agreed or negotiated. I need to check the contract provided to see if there's any mention of this. Looking through the contrac …

**doc 398 · nda-5** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Predicted evidence:* (b) Seller will keep confidential and will not for a period of three (3) years after the execution of this Agreement, directly or indirectly, divulge to anyone, use or otherwise appropriate any of the Confidential Information for any reason or purpose whatsoever except to authorized representatives of Purchasers or the …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party (which I assume refers to Purchasers or Parent in the contract) can share some Confidential Information with their employees. First, I need to look through the contract, specifically the s …

**doc 44 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Attendee agrees that money damages will not be a sufficient remedy for any breach of this Agreement by him and that in addition to all other remedies which may be available, BOMI will be entitled to specific performance and injunctive or other equitable relief, without bond, as a remedy for any such breach or threatene …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party (which would be the Attendee in this contract) may retain some Confidential Information even after returning or destroying it. First, I need to check the contract terms. The contract is be …

**doc 28 · nda-11** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.  
*Predicted evidence:* 9. ... (e) not manufacture, sell, use, or have made devices embodying or utilizing Confidential Information, except as otherwise agreed to by the Parties in writing.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that the Receiving Party shall not reverse engineer any objects which embody the Disclosing Party's Confidential Information. First, I need to look through the contract to f …

**doc 14 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as provided in this Confidentiality Agreement to end; provided that in no event shall stockholder, directly  …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract allows the Receiving Party (which would be the Stockholder here) to share Confidential Information with third parties like consultants, agents, or professional advisors. Looking at the contract,  …

**doc 54 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Any and all documents containing confidential information, copies of such documents and extracts the DISCLOSEE makes from such documents and models the DISCLOSER provides must be returned or demonstrably destroyed and their destruction be confirmed in writing should the DISCLOSER so require.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (DISCLOSEE) can retain some Confidential Information even after returning or destroying it, as per the contract. First, I need to look through the contract terms. The key  …

**doc 456 · nda-3** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Predicted evidence:* Confidential Information shall mean any and all information, whether written, oral, or in any other form, that is disclosed by one party to the other...  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that Confidential Information may include verbally conveyed information. I need to check the contract provided. Looking at the CONTRACT, specifically Attachment A, Section A …

**doc 13 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* That portion of the Confidential Information that may be found in analyses, compilations, studies or other documents prepared by or for a Party and all Confidential Information that is oral will be kept by a Party subject to the terms of this Agreement or destroyed.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information. I need to check the contract provided to see if there's any ment …

**doc 590 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* If either party hereto shall determine that it does not wish to proceed with the Proposed Transaction, such party shall promptly advise the other party of that decision. In that case, or if the Proposed Transaction otherwise is not consummated for any reason, Receiving Party shall, upon Disclosing Party’s written reque …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party must destroy or return some Confidential Information upon termination of the Agreement. I need to check the contract provided. Looking through the sections, I remember there's a section ab …

**doc 16 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Notes and other documents referencing or relating to Confidential Information may be made and kept by the Recipient, but shall continue to be governed by this Confidentiality Agreement until they are destroyed.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (Recipient) can retain some Confidential Information even after returning or destroying it. The contract in question is a Confidentiality and Non-Disclosure Agreement betw …

**doc 69 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* Promptly following the request of the Disclosing Party, the Receiving Party shall, as directed by the Disclosing Party, (i) destroy (and confirm such destruction in writing to the Disclosing Party) or (ii) return all Information it receives or obtains from a Disclosing Party and all copies thereof, as well as all other …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party must destroy or return some Confidential Information upon termination of the Agreement. First, I need to look through the contract to find relevant sections. The question  …

**doc 406 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* All Proprietary Information furnished by Health Card to Ciufo is considered loaned for use solely in connection with the Project, and shall be returned by Ciufo to Health Card upon request by Health Card. Ciufo shall certify that it has destroyed or returned all copies of the Proprietary Information in its possession.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (which would be Ciufo in this contract) may retain some Confidential Information even after returning or destroying it. First, I need to look through the contract terms. T …

**doc 28 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* 6. All Confidential Information disclosed under this Agreement ... shall be returned to the Owner promptly upon written request or the termination or expiration of this Agreement, and shall not thereafter be retained in any form by the Recipient ... The Recipient shall also certify to the Owner the timely destruction o …  
*Model said:* <think> Okay, let's see. The user is asking whether some obligations of the agreement may survive termination. I need to check the contract provided to see if there's any mention of obligations surviving after termination. Looking through the sections, section …

**doc 32 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* The receiving party agrees to immediately return and to deliver all originals and copies of any Confidential Information in its possession or subject to its control to the disclosing party at the address first written above upon the first to occur of the following: termination or expiration of this Agreement; or comple …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party can retain some confidential information even after returning or destroying it, based on the provided contract. First, I need to look through the contract terms. The key s …

**doc 405 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* You and your representatives will not use any of the Evaluation Materials for any purpose other than the exclusive purpose of evaluating a Possible Transaction.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to understand what the contract says about rights to C …

**doc 71 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Prospect and Prospect’s Broker will immediately return to Listing Broker any information and materials provided by Listing Broker.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to retain some Confidential Information even after returning or destroying it. First, I need to look through the contract terms. The key sections here  …

**doc 482 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* For a period of five (5) years following the date of each such disclosure or development the receiving Party shall maintain such Confidential Information as confidential...  
*Model said:* <think> Okay, let's see. The user is asking whether some obligations of the agreement may survive termination. The contract in question is a mutual non-disclosure agreement (NDA) between two companies. First, I need to check if the contract mentions anything a …


### Qwen3-4B (A) — 176 cases, showing 20

**doc 64 · nda-10** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* Receiving Party shall not, without the prior written consent of the Delivering Party, (except as such disclosure is required by law, regulation, government regulatory authority, court order or other legal process, and then only strictly pursuant to the terms of this Confidentiality Agreement), disclose the Information  …  
*Model said:* {"verdict": "Entailment", "evidence": ["Receiving Party shall not, without the prior written consent of the Delivering Party, (except as such disclosure is required by law, regulation, government regulatory authority, court order or other legal process, and th …

**doc 33 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 3. KEEPING CONFIDENTIAL INFORMATION CONFIDENTIAL Each party shall in relation to the Confidential Information of the other party exercise no lesser security measures and degree of care than those which it applies to its own confidential information and which it warrants as providing adequate protection against any unau …  
*Model said:* {"verdict": "Entailment", "evidence": ["3. KEEPING CONFIDENTIAL INFORMATION CONFIDENTIAL\nEach party shall in relation to the Confidential Information of the other party exercise no lesser security measures and degree of care than those which it applies to its …

**doc 9 · nda-1** — gold `NotMentioned`, predicted `Entailment`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* VENDOR, its employees and/or representatives, hold the Confidential Information in trust for BPS’s benefit, and shall, in addition to their respective obligations hereunder, use best efforts at all times to adopt and follow procedures and practices to protect the confidentiality of the Confidential Information and prev …  
*Model said:* {"verdict": "Entailment", "evidence": ["VENDOR, its employees and/or representatives, hold the Confidential Information in trust for BPS’s benefit, and shall, in addition to their respective obligations hereunder, use best efforts at all times to adopt and fol …

**doc 381 · nda-3** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Predicted evidence:* Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including without limitation, proprietary information of the Company, as it exists from time to time (collectively, "Confidential Information"), are …  
*Model said:* {"verdict": "Entailment", "evidence": ["Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including without limitation, proprietary inform …

**doc 14 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as provided in this Confidentiality Agreement to end; provided that in no event shall stockholder, directly  …  
*Model said:* {"verdict": "Contradiction", "evidence": ["The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as pr …

**doc 9 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Nothing in this Agreement may be construed to allow any Party to maintain, use, disclose, or share Confidential Information in a manner not allowed under federal or state law or regulation, or outside the scope of this agreement.  
*Model said:* {"verdict": "Contradiction", "evidence": ["Nothing in this Agreement may be construed to allow any Party to maintain, use, disclose, or share Confidential Information in a manner not allowed under federal or state law or regulation, or outside the scope of thi …

**doc 32 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* 7.1 Unless terminated earlier by mutual agreement of the parties or terminated by Champion in the event of breach by Company of any obligation set forth herein, the term of this Agreement shall continue until the completion of the purpose set forth in Section 2. All Confidential Information shall remain subject to the  …  
*Model said:* { "verdict": "Entailment", "evidence": [ "7.1 Unless terminated earlier by mutual agreement of the parties or terminated by Champion in the event of breach by Company of any obligation set forth herein, the term of this Agreement shall continue until the compl …

**doc 33 · nda-1** — gold `NotMentioned`, predicted `Entailment`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* 1.1 In this Agreement: "Confidential Information" means any information of a confidential nature relating to the Disclosing Party or a member of its Group disclosed (whether before or after the date of this Agreement and whether in writing, verbally or by any other means and whether directly or indirectly) by the Discl …  
*Model said:* {"verdict": "Entailment", "evidence": ["1.1 In this Agreement: \"Confidential Information\" means any information of a confidential nature relating to the Disclosing Party or a member of its Group disclosed (whether before or after the date of this Agreement a …

**doc 45 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* During this period the receiving party shall not disclose the received "Confidential Information" to any third party, and shall not use the received "Confidential Information" for any purpose other than the aforesaid purposes. The receiving party further agrees to use the same degree of care, but no less than a reasona …  
*Model said:* {"verdict": "Entailment", "evidence": ["During this period the receiving party shall not disclose the received \"Confidential Information\" to any third party, and shall not use the received \"Confidential Information\" for any purpose other than the aforesaid …

**doc 605 · nda-7** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made available to Receiving Party, except, in each case, to the extent that Receiving Party is requested pursu …  
*Model said:* {"verdict": "Entailment", "evidence": ["Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made avail …

**doc 405 · nda-15** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* You and your representatives will not use any of the Evaluation Materials for any purpose other than the exclusive purpose of evaluating a Possible Transaction.  
*Model said:* {"verdict": "Entailment", "evidence": ["You and your representatives will not use any of the Evaluation Materials for any purpose other than the exclusive purpose of evaluating a Possible Transaction.", "You and your representatives will keep the Evaluation Ma …

**doc 27 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Recipients will not disclose, distribute, copy or use any Confidential Information or the fact that a Transaction is being considered or is under a production contract without the prior written consent of the other Party and then only to the extent specified in such consent. Confidential Information may be used only to …  
*Model said:* {"verdict": "Entailment", "evidence": ["Recipients will not disclose, distribute, copy or use any Confidential Information or the fact that a Transaction is being considered or is under a production contract without the prior written consent of the other Party …

**doc 70 · nda-15** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* I understand that access to health information created, received, or maintained by FAU or its affiliates is limited to those who have a valid business, medical, or professional need to know the information.  
*Model said:* {"verdict": "Entailment", "evidence": ["I understand that access to health information created, received, or maintained by FAU or its affiliates is limited to those who have a valid business, medical, or professional need to know the information.", "I understa …

**doc 56 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* To return promptly to the disclosing party, or to destroy any copies of such Confidential Information in written, graphic or other tangible form at that Party’s request and any such destruction shall be certified in writing to the disclosing Party by an authorised officer of the Recipient;  
*Model said:* {"verdict": "Entailment", "evidence": ["To return promptly to the disclosing party, or to destroy any copies of such Confidential Information in written, graphic or other tangible form at that Party’s request and any such destruction shall be certified in writ …

**doc 82 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 1.2 shall not use any Confidential Information for your own purposes or to obtain a commercial, trading or other advantage, but shall use such Confidential Information for the sole purpose of evaluating and negotiating the proposed sale of the Business, and, save as set out above, shall not permit any Confidential Info …  
*Model said:* {"verdict": "Contradiction", "evidence": ["1.2 shall not use any Confidential Information for your own purposes or to obtain a commercial, trading or other advantage, but shall use such Confidential Information for the sole purpose of evaluating and negotiatin …

**doc 35 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 2.2 During the course of the proposed discussions, the Parties anticipate that the Disclosing Party will disclose Confidential Information to the Receiving Party, which the Parties agree shall be used solely for the purpose of evaluating the Transaction.  
*Model said:* {"verdict": "Contradiction", "evidence": ["2.2 During the course of the proposed discussions, the Parties anticipate that the Disclosing Party will disclose Confidential Information to the Receiving Party, which the Parties agree shall be used solely for the p …

**doc 10 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Upon the Disclosing Party’s written request, the Receiving Party shall promptly return to the Disclosing Party all Confidential Information in its possession or control, and promptly destroy or erase all notes, memoranda and other material prepared by the Receiving Party which reflect, interpret, evaluate, include or a …  
*Model said:* {"verdict": "Entailment", "evidence": ["Upon the Disclosing Party’s written request, the Receiving Party shall promptly return to the Disclosing Party all Confidential Information in its possession or control, and promptly destroy or erase all notes, memoranda …

**doc 582 · nda-18** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not solicit some of Disclosing Party's representatives.  
*Predicted evidence:* 8. From the date of this Agreement until the earlier to occur of (a) 11:59 p.m. (New York time) on February 3, 2011 and (b) the execution of a definitive agreement between the Company and you relating to a Transaction, except for discussions with you regarding the Transaction, the Company and its affiliates shall not,  …  
*Model said:* {"verdict": "Entailment", "evidence": ["8. From the date of this Agreement until the earlier to occur of (a) 11:59 p.m. (New York time) on February 3, 2011 and (b) the execution of a definitive agreement between the Company and you relating to a Transaction, e …

**doc 9 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* VENDOR, its employees and/or representatives, hold the Confidential Information in trust for BPS’s benefit, and shall, in addition to their respective obligations hereunder, use best efforts at all times to adopt and follow procedures and practices to protect the confidentiality of the Confidential Information and prev …  
*Model said:* {"verdict": "Entailment", "evidence": ["VENDOR, its employees and/or representatives, hold the Confidential Information in trust for BPS’s benefit, and shall, in addition to their respective obligations hereunder, use best efforts at all times to adopt and fol …

**doc 507 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* In consideration of Big Sky's disclosure to Holdings of the Information, Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectiv …  
*Model said:* {"verdict": "Contradiction", "evidence": ["In consideration of Big Sky's disclosure to Holdings of the Information, Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky,  …


### Qwen3-4B (B) — 102 cases, showing 20

**doc 590 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 1.1 "Proprietary Information" ... (collectively, "Derived Information"), whether tangible or intangible, furnished or prepared in writing or in oral, graphic, electronic or any other form or manner.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to create a copy of some Confidential Information under certain circumstances. The answer needs to be either Entailment, Contradiction, or NotMentioned …

**doc 56 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* 3. Each Party acknowledges and agrees: f. To return promptly to the disclosing party, or to destroy any copies of such Confidential Information in written, graphic or other tangible form at that Party’s request and any such destruction shall be certified in writing to the disclosing Party by an authorised officer of th …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party shall destroy or return some Confidential Information upon the termination of the Agreement. First, I need to look through the cont …

**doc 39 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 1. Purpose ... for the sole purpose of compiling and submitting a bid in response to the County’s Invitation for Bid to construct the Project.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …

**doc 69 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* Promptly following the request of the Disclosing Party, the Receiving Party shall, as directed by the Disclosing Party, (i) destroy (and confirm such destruction in writing to the Disclosing Party) or (ii) return all Information it receives or obtains from a Disclosing Party and all copies thereof, as well as all other …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party shall destroy or return some Confidential Information upon the termination of the Agreement. The answer needs to be either Entailme …

**doc 405 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Upon the Company's request, you shall promptly deliver to the Company or destroy all written Evaluation Materials and any other written materials without retaining, in whole or in part, any copies, extracts or other reproductions (whatever the form or storage medium) of such materials, and shall certify the destruction …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to retain some Confidential Information even after returning or destroying it. The answer needs to be based on the provided contract text. First, I nee …

**doc 411 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any business contacts, customers, suppliers, technology …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party may create a copy of some Confidential Information in some circumstances. The answer needs to be either Entailment, Contradiction,  …

**doc 69 · nda-18** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not solicit some of Disclosing Party's representatives.  
*Predicted evidence:* 7. Non-Solicitation. The Company acknowledges that Chief Enterprises has invested and will continue to invest significant time and money to develop valuable, continuing relationships with existing and prospective clients and customers of Chief Enterprises. The Company further recognizes that it will obtain valuable Con …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party shall not solicit some of the Disclosing Party's representatives. The answer needs to be either Entailment, Contradiction, or NotMe …

**doc 32 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* All Confidential Information shall remain subject to the obligations to maintain the confidentiality thereof indefinitely following the date of termination of this Agreement.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that some obligations may survive termination. The answer options are Entailment, Contradiction, or NotMentioned. First, I need to look through the contract …

**doc 17 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 3k. Upon completion or termination of the contract for any reason, the contractor will immediately deliver all VA records, data, copies of VA records and data, software and equipment, and information about VA data and systems recorded or documented by the contractor, in its possession or the possession of any subcontra …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which I assume is the contractor) to retain some Confidential Information even after the return or destruction of that information. First, I need to l …

**doc 590 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* If either party hereto shall determine that it does not wish to proceed with the Proposed Transaction, such party shall promptly advise the other party of that decision. In that case, or if the Proposed Transaction otherwise is not consummated for any reason, Receiving Party shall, upon Disclosing Party’s written reque …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party shall destroy or return some Confidential Information upon the termination of the Agreement. The answer needs to be either Entailme …

**doc 70 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my possession. I understand that my confidentiality obligations under this Agreement will continue after the …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to retain some Confidential Information even after returning or destroying it. The contract in question is the Confidentiality, Non-Disclosure & Approp …

**doc 10 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 2.01 Confidential Information: "Confidential Information" means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party that is clearly marked "Confidential" or clearly intended to be disclosed in confidence.", "2.03 Kept in Co …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the agreement does not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or Not …

**doc 482 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 3. The receiving Party shall not, without prior written permission of the disclosing Party, furnish to any third Party any information, equipment or material embodying or made by use of any Confidential Information received or developed hereunder nor use such information for purposes other than internal evaluation so l …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the receiving party to retain some confidential information even after returning or destroying it. The answer needs to be based on the provided contract text. First, I nee …

**doc 79 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 4. Ownership/Return of Confidential Information. Dealer agrees and acknowledges that the Confidential Information is the property of CIF, and upon termination of this Agreement all rights to the use and ownership of the Confidential Information shall be retained exclusively by CIF.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (Dealer) to retain some Confidential Information even after returning or destroying it. The answer needs to be either Entailment, Contradiction, or Not …

**doc 14 · nda-3** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Predicted evidence:* The Stockholder will not disclose, publish, or communicate Confidential Material to any person or entity in any form (written, oral, electronic or through any other medium), either directly or indirectly, except as provided in this Confidentiality Agreement to end; provided that in no event shall stockholder, directly  …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that Confidential Information may include verbally conveyed information. The answer needs to be either Entailment, Contradiction, or NotMentioned, with evid …

**doc 398 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* Seller will keep confidential and will not for a period of three (3) years after the execution of this Agreement, directly or indirectly, divulge to anyone, use or otherwise appropriate any of the Confidential Information for any reason or purpose whatsoever except to authorized representatives of Purchasers or the Par …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which I assume is Seller here) to share some Confidential Information with third parties like consultants, agents, or advisors. The answer needs to be …

**doc 12 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 1.1 With the understanding that CLIENT’s Formulations and Processing Procedures are proprietary to CLIENT, to the extent that CLIENT shall disclose to BROOKS personal, financial and product information, including their Formulations and Processing Procedures (“Confidential Information”), then all the Confidential Inform …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement does not grant the Receiving Party (which would be BROOKS in this case) any right to Confidential Information. The answer needs to be eit …

**doc 586 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* Section 6. Return and Destruction of Evaluation Material. At any time after termination of discussions by either Party with respect to the Transaction, upon the request of the Furnishing Party, the Receiving Party shall promptly (and in no event later than five business days after such request) (a) redeliver or cause t …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party shall destroy or return some Confidential Information upon the termination of the Agreement. The answer needs to be either Entailme …

**doc 35 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 4.2.3 not use, exploit or in any other manner whatsoever apply or reproduce the Confidential Information except to the extent required to achieve the Purpose  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to create a copy of some Confidential Information under certain circumstances. The answer needs to be either Entailment, Contradiction, or NotMentioned …

**doc 17 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 3.b. Any VA information, software, applications, computer systems, and hardware accessed by the contractor in the performance of the contract remain the sole property of VA.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or No …


### Qwen3-8B (A) — 290 cases, showing 20

**doc 411 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any business contacts, customers, suppliers, technology …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business …

**doc 71 · nda-12** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluation.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluation …

**doc 56 · nda-10** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* Neither disclosure of Confidential Information, nor this Agreement shall be construed as (i) requiring a Party to disclose or accept Confidential Information, (ii) permitting a Party to use or sell the Confidential Information or products derived therefrom, or (iii) granting to either Party any rights by license or oth …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Neither disclosure of Confidential Information, nor this Agreement shall be construed as (i) requiring a Party to disclose or accept Confidential Information, (ii) permitting a Party to use or sell the Confidential  …

**doc 456 · nda-5** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Predicted evidence:* No rights or obligations other than expressly recited herein are to be implied here from. Nothing except that expressly stated herein shall affect either party's present or prospective rights under any country's patent laws, or be construed as granting any license under any present or future patent or application there …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "No rights or obligations other than expressly recited herein are to be implied here from. Nothing except that expressly stated herein shall affect either party's present or prospective rights under any country's pat …

**doc 563 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* The Recipient will, within five business days of a written request by the Disclosing Party, confirm that it has destroyed the original provided to Recipient (and all copies made by it) any documents (in whatever form or medium) provided, or made available for review by, the Disclosing Party (including any Derivative Co …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Recipient will, within five business days of a written request by the Disclosing Party, confirm that it has destroyed the original provided to Recipient (and all copies made by it) any documents (in whatever form o …

**doc 29 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Recipient will not reproduce, use, disseminate, or disclose CI to any person or entity, except to its employees and authorized representatives (i.e., temporary employees, consultants, and contractors) who need to know the CI to further the Purpose and are bound by confidentiality obligations at least as restrictive as  …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Recipient will not reproduce, use, disseminate, or disclose CI to any person or entity, except to its employees and authorized representatives (i.e., temporary employees, consultants, and contractors) who need to know  …

**doc 486 · nda-15** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and know-how, and will disclose such information only pursuant to the express written permission of the party …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and kno …

**doc 586 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Notwithstanding the return, destruction or erasure of Evaluation Material hereunder, the Receiving Party and its Representatives shall continue to be bound by their confidentiality and other obligations hereunder.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Notwithstanding the return, destruction or erasure of Evaluation Material hereunder, the Receiving Party and its Representatives shall continue to be bound by their confidentiality and other obligations hereunder." ] }

**doc 605 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* This Agreement and the obligations of the parties hereunder (other than Section 4(b)) shall terminate upon the earlier to occur of (i) twenty (20) days from the date of the Refinancing Meeting, (ii) such time as Receiving Party is no longer in possession of Blueknight Non-public Information and (iii) with respect to an …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "This Agreement and the obligations of the parties hereunder (other than Section 4(b)) shall terminate upon the earlier to occur of (i) twenty (20) days from the date of the Refinancing Meeting, (ii) such time as Rec …

**doc 17 · nda-10** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* The contractor may disclose VA sensitive information to persons other than in the performance of this contract as authorized by the contract and this agreement in only two situations: (1) pursuant to an order of a court of competent jurisdiction; or (2) with VA's prior written authorization.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The contractor may disclose VA sensitive information to persons other than in the performance of this contract as authorized by the contract and this agreement in only two situations: (1) pursuant to an order of a c …

**doc 10 · nda-12** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* The Receiving Party shall not, in any manner, derive any benefit, directly or indirectly, from the Confidential Information or the use of such Confidential Information, for any purpose.  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Receiving Party shall not, in any manner, derive any benefit, directly or indirectly, from the Confidential Information or the use of such Confidential Information, for any purpose." ] }

**doc 39 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s dut …

**doc 54 · nda-10** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* The DISCLOSEE agrees to keep any and all written, electronic and oral information and models conveyed or made available to it directly or indirectly in the course of the collaboration strictly in confidence and not to disclose it or make it available to any third party without the DISCLOSER's consent in writing.  
*Model said:* { "verdict": "Entailment", "evidence": [ "The DISCLOSEE agrees to keep any and all written, electronic and oral information and models conveyed or made available to it directly or indirectly in the course of the collaboration strictly in confidence and not to  …

**doc 35 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* 6.1 The Disclosing Party may, at any time, request the Receiving Party to return any material containing, pertaining to or relating to Confidential Information disclosed by the Disclosing Party, its Affiliates or any of their respective employees, representatives, officers, directors or advisers, in relation to the Tra …  
*Model said:* { "verdict": "Entailment", "evidence": [ "6.1 The Disclosing Party may, at any time, request the Receiving Party to return any material containing, pertaining to or relating to Confidential Information disclosed by the Disclosing Party, its Affiliates or any o …

**doc 406 · nda-15** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Ciufo (a) must receive and hold the Proprietary Information in trust and in strictest confidence;  
*Model said:* { "verdict": "Entailment", "evidence": [ "Ciufo (a) must receive and hold the Proprietary Information in trust and in strictest confidence;" ] }

**doc 64 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Non-tangible information will be Information only to the extent that it is first disclosed in non-tangible form and is orally identified at the time of disclosure as confidential information and is summarized in tangible form conspicuously marked as “confidential information” and delivered to Receiving Party in such ta …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Non-tangible information will be Information only to the extent that it is first disclosed in non-tangible form and is orally identified at the time of disclosure as confidential information and is summarized in tangib …

**doc 582 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* The term "Confidential Information" does not include any information which (a) at the time of disclosure or thereafter is generally available to or known by the public (other than as a result of its disclosure by you or your Representatives in breach of this Agreement),  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The term \"Confidential Information\" does not include any information which (a) at the time of disclosure or thereafter is generally available to or known by the public (other than as a result of its disclosure by  …

**doc 13 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Once the Party no longer requires the use of such information for the purpose of performing its Transmission Functions, the Confidential Information shall be returned or destroyed in accordance with this Agreement.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Once the Party no longer requires the use of such information for the purpose of performing its Transmission Functions, the Confidential Information shall be returned or destroyed in accordance with this Agreement." …

**doc 598 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* Confidential Information includes any reports, analyses or notes that are based on, reflect or contain Confidential Information.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidential Information includes any reports, analyses or notes that are based on, reflect or contain Confidential Information." ] }

**doc 605 · nda-7** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made available to Receiving Party, except, in each case, to the extent that Receiving Party is requested pursu …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made ava …


### Qwen3-8B (B) — 127 cases, showing 20

**doc 61 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 5. Except as otherwise provided in any future agreement, at any time, Company agrees that it shall return to the AOC upon request all documents, records, notebooks, computer media or other stored information of any form or type whatsoever containing any Confidential Information, including all copies thereof, then in it …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party (which would be the Company in this case) may retain some Confidential Information even after returning or destroying it. First, I  …

**doc 582 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 5. The Company may elect at any time to terminate further access by you to the Confidential Information. Following any written request by the Company, you agree, at the Company’s election, to either promptly redeliver to the Company or destroy all written Confidential Information in your possession or your Representati …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which I assume is Verizon in this case) to create copies of Confidential Information under certain circumstances. First, I need to look through the co …

**doc 590 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 2.1 Non-Disclosure. ... shall not disclose or reveal, and shall cause its Representatives not to disclose or reveal, in any manner whatsoever, in whole or in part, any Proprietary Information ... Receiving Party shall not, and shall cause its Representatives to not, use any Proprietary Information for any purpose other …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may create a copy of some Confidential Information under the given contract. The answer needs to be based on the contract provided, and I have to determine if it's an Enta …

**doc 12 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 3.6 Discloses Confidential Information to others which necessarily results from performing the services has contracted with BROOKS to perform.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (BROOKS) to create copies of Confidential Information under certain circumstances. First, I need to look through the contract to find relevant sections …

**doc 19 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 4. Return of Confidential Information: The Confidant shall immediately on notice at any time from City return to City, or destroy, any and all Confidential Information in accordance with City’s direction.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information." First, I need to understa …

**doc 411 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* 1. Non-Competition. Each of the Stockholders severally agrees that for a period (the "Restrictive Period") which shall commence on the Effective Date of the Merger and shall expire on a date which shall be the latest to occur of: (a) such date as such Stockholder shall no longer be employed on a full-time basis with AU …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Some obligations of Agreement may survive termination of Agreement." First, I need to understand what the question is really about. The key here is d …

**doc 13 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 8. Ownership and Use of Confidential Information. All Confidential Information delivered by a Disclosing Party to a Party pursuant to this Agreement shall be and remain the property of the Disclosing Party, and such Confidential Information shall be promptly returned to the Disclosing Party upon request. Once the Party …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information. First, I need to underst …

**doc 32 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* The receiving party shall not disclose, disseminate, discuss or communicate any part or all of the Confidential Information to any third party, person or entity.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party may create a copy of some Confidential Information in some circumstances, based on the provided contract. First, I need to look through the contract to find sections relat …

**doc 590 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* Receiving Party shall, upon Disclosing Party’s written request, promptly deliver to Disclosing Party all Proprietary Information, and, at Disclosing Party’s sole election, return or destroy (provided that such destruction shall be certified by a duly authorized Representative of Receiving Party) all copies, reproductio …  
*Model said:* <think> Okay, let's tackle this question. The user wants to know if the contract supports the statement that "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement." First, I need to look through the contract t …

**doc 19 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* 4. Return of Confidential Information: The Confidant shall immediately on notice at any time from City return to City, or destroy, any and all Confidential Information in accordance with City’s direction.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement." First, I need to look through the contract  …

**doc 482 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* 2. For a period of five (5) years following the date of each such disclosure or development the receiving Party shall maintain such Confidential Information as confidential...  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Some obligations of Agreement may survive termination of Agreement." First, I need to recall what the contract says. The contract is a Non-Disclosure …

**doc 75 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* The Investor acknowledges and agrees that the Confidential Information is strictly confidential and is intended solely for the Investor’s limited use and benefit in determining whether the Investor desires to express any further interest in the acquisition and development of the Property.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports, contradicts, or doesn't mention the statement: "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to understand what the c …

**doc 440 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* 6. If any party provides any of the other parties with written notification of its request for return of its Confidential Information, then the other party(ies) shall promptly return (unless destroyed pursuant to the immediately following sentence) to the notifying party all materials and information comprising the not …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party must destroy or return some Confidential Information upon termination of the Agreement. First, I need to look through the contract  …

**doc 79 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* 4. Ownership/Return of Confidential Information. Dealer agrees and acknowledges that the Confidential Information is the property of CIF, and upon termination of this Agreement all rights to the use and ownership of the Confidential Information shall be retained exclusively by CIF. In the event that CIF requests that D …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Some obligations of Agreement may survive termination of Agreement." First, I need to recall what the contract says. The user provided a Confidential …

**doc 486 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* d) This Agreement and each additional agreement concluded or written or verbal disclosure made between the Parties, shall be kept confidential and is not to be reproduced, communicated or distributed in any manner whatsoever except on a "need to know" basis to persons directly involved with the closing of any transacti …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to share Confidential Information with third parties like consultants, agents, and professional advisors. First, I need to look through the contract's  …

**doc 54 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* 7. As with the confidential information, no results obtained from that information may be disclosed to any third parties or made available otherwise without the DISCLOSER's consent in writing. This also applies without limitations to any possible termination of the business relationship between both parties.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Some obligations of Agreement may survive termination of Agreement." First, I need to recall what the contract says. The user provided a Non-Disclosu …

**doc 61 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* 5. Except as otherwise provided in any future agreement, at any time, Company agrees that it shall return to the AOC upon request all documents, records, notebooks, computer media or other stored information of any form or type whatsoever containing any Confidential Information, including all copies thereof, then in it …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party (which would be the Company in this case) must destroy or return some Confidential Information upon termination of the Agreement. F …

**doc 10 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* ARTICLE 2.07 Return of Confidential Information: Upon the Disclosing Party’s written request, the Receiving Party shall promptly return to the Disclosing Party all Confidential Information in its possession or control, and promptly destroy or erase all notes, memoranda and other material prepared by the Receiving Party …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party must destroy or return some Confidential Information upon the termination of the Agreement. First, I need to look through the contr …

**doc 12 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* 1.1 With the understanding that CLIENT’s Formulations and Processing Procedures are proprietary to CLIENT, to the extent that CLIENT shall disclose to BROOKS personal, financial and product information, including their Formulations and Processing Procedures (“Confidential Information”), then all the Confidential Inform …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports, contradicts, or doesn't mention the statement: "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to understand what the c …

**doc 440 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* 6. If any party provides any of the other parties with written notification of its request for return of its Confidential Information, then the other party(ies) shall promptly return (unless destroyed pursuant to the immediately following sentence) to the notifying party all materials and information comprising the not …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Some obligations of Agreement may survive termination of Agreement." First, I need to recall what the contract says. The contract is a Non-Disclosure …


### gemma-4-12B-it (A) — 128 cases, showing 20

**doc 563 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Notwithstanding the immediately preceding sentence, the parties agree that destruction will not be made of (i) any electronic copy of any Confidential Information that was created pursuant to the standard electronic backup and archival procedures of the Recipient if (x) personnel of the Recipient whose functions are no …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Notwithstanding the immediately preceding sentence, the parties agree that destruction will not be made of (i) any electronic copy of any Confidential Information that was created pursuant to the standard electro …

**doc 547 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* The Receiving Party will disclose Confidential Information only to its employees and to its representative whose participation is necessary or helpful to evaluate and advance to the Business Transaction and who have agreed to be bound by the terms of this Agreement.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The Receiving Party will disclose Confidential Information only to its employees and to its representative whose participation is necessary or helpful to evaluate and advance to the Business Transaction and wh …

**doc 79 · nda-3** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Predicted evidence:* 1. Confidential Information. The term "Confidential Information" shall mean and include that technical and non-technical information disclosed by CIF to Dealer relating to: (a) CIF's website, information system, and related software and discs, and any data and information contained therein, (b) information and specific …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["1. Confidential Information. The term \"Confidential Information\" shall mean and include that technical and non-technical information disclosed by CIF to Dealer relating to: (a) CIF's website, information system …

**doc 152 · nda-10** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* Neither party shall make or permit others to make any reference to the subject matter of the Agreement, or the Information or use the name of the other party in any public announcements, promotional, marketing or sales materials or efforts without the prior written consent of the other party and such consent shall not  …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Neither party shall make or permit others to make any reference to the subject matter of the Agreement, or the Information or use the name of the other party in any public announcements, promotional, marketing or …

**doc 64 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* The Receiving Party shall keep a record of the written Information furnished to the Receiving Party and of the location of all such Information, including all photocopies or other reproductions, a copy of which record shall be made available to the Delivering Party upon written request.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["The Receiving Party shall keep a record of the written Information furnished to the Receiving Party and of the location of all such Information, including all photocopies or other reproductions, a copy of which r …

**doc 398 · nda-12** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* Seller acknowledges and agrees that these prohibitions against disclosure of Confidential Information are in addition to, and not in lieu of, any rights or remedies that Purchasers or Parent may have available pursuant to the laws of any jurisdiction or at common law to prevent disclosure of trade secrets or proprietar …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Seller acknowledges and agrees that these prohibitions against disclosure of Confidential Information are in addition to, and not in lieu of, any rights or remedies that Purchasers or Parent may have available pu …

**doc 19 · nda-1** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Predicted evidence:* c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RFP and/or the Permitted Use, which comes into the possession or knowledge of the Confidant, including …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the C …

**doc 29 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed in non-tangible form that is identified as confidential at the time of disclosure and summarized in a writing l …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["CI means Discloser’s non-public written information, in any form, and all copies, summaries and extracts which is identified in writing as confidential at the time of disclosure, and any information disclosed  …

**doc 12 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* Upon termination of the business relationship between the Parties, BROOKS shall, upon written request of CLIENT, deliver any records, data, information and other documents, and all copies thereof, furnished by CLIENT to BROOKS.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Upon termination of the business relationship between the Parties, BROOKS shall, upon written request of CLIENT, deliver any records, data, information and other documents, and all copies thereof, furnished by CL …

**doc 610 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* Following written notice by one party to another of the termination of discussions between the parties, upon written request of the disclosing party, the receiving party shall take reasonable steps to instruct all persons involved in the Transaction to destroy all Confidential Information furnished to the receiving par …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Following written notice by one party to another of the termination of discussions between the parties, upon written request of the disclosing party, the receiving party shall take reasonable steps to instruct al …

**doc 56 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* To return promptly to the disclosing party, or to destroy any copies of such Confidential Information in written, graphic or other tangible form at that Party’s request and any such destruction shall be certified in writing to the disclosing Party by an authorised officer of the Recipient;  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["To return promptly to the disclosing party, or to destroy any copies of such Confidential Information in written, graphic or other tangible form at that Party’s request and any such destruction shall be certified …

**doc 10 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* "Confidential Information" means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party that is clearly marked “Confidential” or clearly intended to be disclosed in confidence.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["\"Confidential Information\" means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party that is clearly marked “C …

**doc 405 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* Upon the Company's request, you shall promptly deliver to the Company or destroy all written Evaluation Materials and any other written materials without retaining, in whole or in part, any copies, extracts or other reproductions (whatever the form or storage medium) of such materials, and shall certify the destruction …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Upon the Company's request, you shall promptly deliver to the Company or destroy all written Evaluation Materials and any other written materials without retaining, in whole or in part, any copies, extracts or …

**doc 20 · nda-20** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* The Receiving Party shall certify in writing that the information has been returned, deleted or destroyed (as applicable) and that no copies have been retained of any such Confidential Information, either partly or wholly.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The Receiving Party shall certify in writing that the information has been returned, deleted or destroyed (as applicable) and that no copies have been retained of any such Confidential Information, either part …

**doc 28 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* Confidential Information may be disclosed in written or other tangible form (including magnetic media) or by oral, visual or other means.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Confidential Information may be disclosed in written or other tangible form (including magnetic media) or by oral, visual or other means.", "Confidential Information disclosed hereunder is intended to be used  …

**doc 398 · nda-5** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Predicted evidence:* Seller will keep confidential and will not for a period of three (3) years after the execution of this Agreement, directly or indirectly, divulge to anyone, use or otherwise appropriate any of the Confidential Information for any reason or purpose whatsoever except to authorized representatives of Purchasers or the Par …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Seller will keep confidential and will not for a period of three (3) years after the execution of this Agreement, directly or indirectly, divulge to anyone, use or otherwise appropriate any of the Confidential In …

**doc 54 · nda-19** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Predicted evidence:* This also applies without limitations to any possible termination of the business relationship between both parties.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["This also applies without limitations to any possible termination of the business relationship between both parties.", "Such persons must demonstrably be bound beforehand to keep the information confidential to t …

**doc 486 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and know-how, and will disclose such information only pursuant to the express written permission of the party …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledg …

**doc 381 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* they will not, and will use their best efforts to ensure that their directors, officers, employees, advisers, agents and consultants do not, disclose any Confidential Information concerning the Company and/or its subsidiaries or affiliates, to any person, firm, corporation, association or other entity, for any reason w …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["they will not, and will use their best efforts to ensure that their directors, officers, employees, advisers, agents and consultants do not, disclose any Confidential Information concerning the Company and/or  …

**doc 81 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “proprietary,” “secret,” or similar legend, and (ii) discussions relating to that information whether those discussio …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is (i) information in tangible form that bears a “confidential,” “propr …


### granite-4.2-8b (A) — 194 cases, showing 20

**doc 456 · nda-20** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Predicted evidence:* 1. Either party may terminate this MNDA without cause upon five (5) days written notice given to the other, provided that confidentiality obligations under Section A of this Attachment A shall survive termination hereof.  
*Model said:* { "verdict": "Entailment", "evidence": [ "1. Either party may terminate this MNDA without cause upon five (5) days written notice given to the other, provided that confidentiality obligations under Section A of this Attachment A shall survive termination hereo …

**doc 70 · nda-13** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Predicted evidence:* I understand that access to health information created, received, or maintained by FAU or its affiliates is limited to those who have a valid business, medical, or professional need to know the information.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "I understand that access to health information created, received, or maintained by FAU or its affiliates is limited to those who have a valid business, medical, or professional need to know the information.", "I wil …

**doc 152 · nda-12** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* 4. With respect to the obligations of the receiving party to the protection and use of the Information, no obligations shall be imposed upon the receiving party with respect to any portion of said Information, which the receiving party can show (a) corresponds in substance to Information in the receiving party’s posses …  
*Model said:* { "verdict": "Entailment", "evidence": [ "4. With respect to the obligations of the receiving party to the protection and use of the Information, no obligations shall be imposed upon the receiving party with respect to any portion of said Information, which th …

**doc 12 · nda-17** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 1.1 With the understanding that CLIENT’s Formulations and Processing Procedures are proprietary to CLIENT, to the extent that CLIENT shall disclose to BROOKS personal, financial and product information, including their Formulations and Processing Procedures (“Confidential Information”), then all the Confidential Inform …  
*Model said:* { "verdict": "Entailment", "evidence": [ "1.1 With the understanding that CLIENT’s Formulations and Processing Procedures are proprietary to CLIENT, to the extent that CLIENT shall disclose to BROOKS personal, financial and product information, including their …

**doc 381 · nda-7** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Predicted evidence:* 2. Protection of Information. Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including without limitation, proprietary information of the Company, as it exists from time to time (collectively, "C …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2. Protection of Information. Selling Parties recognize and acknowledge that the Company's trade secrets and all other confidential and proprietary information of a business, financial or other nature, including wit …

**doc 54 · nda-10** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* 3. The DISCLOSEE agrees to keep any and all written, electronic and oral information and models conveyed or made available to it directly or indirectly in the course of the collaboration strictly in confidence and not to disclose it or make it available to any third party without the DISCLOSER's consent in writing.  
*Model said:* { "verdict": "Entailment", "evidence": [ "3. The DISCLOSEE agrees to keep any and all written, electronic and oral information and models conveyed or made available to it directly or indirectly in the course of the collaboration strictly in confidence and not  …

**doc 17 · nda-15** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* b. Any VA information, software, applications, computer systems, and hardware accessed by the contractor in the performance of the contract remain the sole property of VA.  
*Model said:* { "verdict": "Entailment", "evidence": [ "b. Any VA information, software, applications, computer systems, and hardware accessed by the contractor in the performance of the contract remain the sole property of VA.", "e. The contractor will not authorize anyone …

**doc 37 · nda-17** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Predicted evidence:* 2.1.2 have in place and maintain proper security measures and procedures to protect the confidentiality of the Confidential Information (having regard to its form and nature);  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2.1.2 have in place and maintain proper security measures and procedures to protect the confidentiality of the Confidential Information (having regard to its form and nature);", "2.1.3 not disclose or permit the dis …

**doc 19 · nda-12** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* Permitted Use means private evaluation by the Confidant solely for the purpose of preparing a submission(s) to City in response to the RFP, and for no other purpose whatsoever.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Permitted Use means private evaluation by the Confidant solely for the purpose of preparing a submission(s) to City in response to the RFP, and for no other purpose whatsoever.", "Confidant may use the Confidential  …

**doc 71 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* In the event the transaction is not successful, Prospect and Prospect’s Broker will immediately return to Listing Broker any information and materials provided by Listing Broker.  
*Model said:* { "verdict": "Entailment", "evidence": [ "In the event the transaction is not successful, Prospect and Prospect’s Broker will immediately return to Listing Broker any information and materials provided by Listing Broker." ] }

**doc 582 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* 1. You hereby agree that the Confidential Information will be used solely for the purpose of evaluating and implementing the Transaction and that such information will be kept confidential by you; provided that any such information may be disclosed to your officers, directors, employees, affiliates, accountants, attorn …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1. You hereby agree that the Confidential Information will be used solely for the purpose of evaluating and implementing the Transaction and that such information will be kept confidential by you; provided that any  …

**doc 71 · nda-15** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluation.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluation …

**doc 71 · nda-13** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Predicted evidence:* Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluation.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Prospect and Prospect’s Broker acknowledge that all information and materials provided by Listing Broker regarding the above-referenced Property is confidential and may not be used for any purpose other than evaluat …

**doc 411 · nda-3** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Predicted evidence:* (a) Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any business contacts, customers, suppliers, techno …  
*Model said:* { "verdict": "Entailment", "evidence": [ "(a) Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other busines …

**doc 20 · nda-10** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Predicted evidence:* 2.4 The Parties shall only use Confidential Information for the purpose of negotiation and/or evaluation of any co-operation between the Parties. The Parties shall not disclose Confidential Information to any third party, other than to their officers, employees or consultants (engaged in the matter of this Agreement) o …  
*Model said:* { "verdict": "Entailment", "evidence": [ "2.4 The Parties shall only use Confidential Information for the purpose of negotiation and/or evaluation of any co-operation between the Parties. The Parties shall not disclose Confidential Information to any third par …

**doc 39 · nda-15** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Predicted evidence:* The purpose of this agreement is to allow the Bidder access to Confidential Information as defined herein detailing the design of the Project for the sole purpose of compiling and submitting a bid in response to the County’s Invitation for Bid to construct the Project.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The purpose of this agreement is to allow the Bidder access to Confidential Information as defined herein detailing the design of the Project for the sole purpose of compiling and submitting a bid in response to the …

**doc 10 · nda-2** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Predicted evidence:* 2.01 Confidential Information "Confidential Information" means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party that is clearly marked “Confidential” or clearly intended to be disclosed in confidence.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2.01 Confidential Information\n\"Confidential Information\" means, subject to section 2.02 below any additional information requested by the Supplier and provided by CPR, or any information supplied by either party  …

**doc 14 · nda-12** — gold `NotMentioned`, predicted `Contradiction`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Predicted evidence:* 1. Stockholder agrees that the Confidential Material is being produced to them only for purposes reasonably related to Stockholder’s interests as a stockholder of the Company and that Stockholder shall not utilize Confidential Material, any other information produced by the Company or anything derived from Confidential …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1. Stockholder agrees that the Confidential Material is being produced to them only for purposes reasonably related to Stockholder’s interests as a stockholder of the Company and that Stockholder shall not utilize C …

**doc 29 · nda-4** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Predicted evidence:* Company and Adobe may make disclosures of CI to each other for discussions, evaluation, or consultations, concerning their prospective or existing business relationship (the Purpose).  
*Model said:* { "verdict": "Entailment", "evidence": [ "Company and Adobe may make disclosures of CI to each other for discussions, evaluation, or consultations, concerning their prospective or existing business relationship (the Purpose).", "Recipient will not reproduce, u …

**doc 73 · nda-16** — gold `NotMentioned`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Predicted evidence:* 6. Return of Confidential Information and Confidential Materials. The Receiving Party shall keep a written record of the Confidential Information and the Confidential Materials furnished to it and the location of such Confidential Information and the Confidential Materials. At any time, at the Disclosing Party’s reques …  
*Model said:* { "verdict": "Entailment", "evidence": [ "6. Return of Confidential Information and Confidential Materials. The Receiving Party shall keep a written record of the Confidential Information and the Confidential Materials furnished to it and the location of such  …


## 6. Verdict right, evidence wrong (the faithfulness failure)

The verdict is correct so the answer looks right, but the cited clause is not the one the annotator marked. Most dangerous failure in a legal product: the citation is what a reviewer trusts.


### Qwen3-14B (A) — 325 cases, showing 20

**doc 45 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* 2. Unless otherwise expressly authorized by the disclosing party, the receiving party agrees to retain the “Confidential Information” in confidence for a period of three (3) years from the date of receipt of the “Confidential Information” or until such time as the information no longer qualifies as “Confidential Inform …  
*Predicted evidence:* Notwithstanding termination or expiration, all obligations of confidentiality shall survive and continue to bind the receiving parties for the period set forth in Section 2.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Notwithstanding termination or expiration, all obligations of confidentiality shall survive and continue to bind the receiving parties for the period set forth in Section 2." ] }

**doc 70 · nda-16** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Gold evidence:*  I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my possession.  
*Predicted evidence:* I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my possession.  
*Model said:* { "verdict": "Entailment", "evidence": [ "I understand that upon termination of my employment/affiliation/association with FAU, I will immediately return or destroy, as appropriate, any FAU Confidential Information, Restricted Data and Sensitive Data in my pos …

**doc 507 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* Notwithstanding the foregoing, the above restrictions shall not apply to the following:  
*Predicted evidence:* disclosures that are required to be made by Holdings or any of its Representatives under legal process by subpoena or other court order or other applicable laws or regulations (provided, however, that Holdings makes reasonable efforts to provide copies of such Information to, or informs Big Sky before disclosure)  
*Model said:* { "verdict": "Entailment", "evidence": [ "disclosures that are required to be made by Holdings or any of its Representatives under legal process by subpoena or other court order or other applicable laws or regulations (provided, however, that Holdings makes re …

**doc 563 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 1. “Confidential Information” means, for each party, whether in oral, written or in electronic form:  
*Predicted evidence:* The Recipient will keep, and will cause its Representatives to keep, the Confidential Information of the Disclosing Party confidential. The Recipient will not disclose the Confidential Information or use or permit the Confidential Information to be disclosed or used, to or by any person except Recipient’s Representativ …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Recipient will keep, and will cause its Representatives to keep, the Confidential Information of the Disclosing Party confidential. The Recipient will not disclose the Confidential Information or use or permit the  …

**doc 72 · nda-16** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Gold evidence:* 11. Upon receipt of a written request to the undersigned, the undersigned and its Representatives shall promptly return to BDO any and all Information received, printed, downloaded or otherwise copied together with personal notes including written materials and any notes of verbal conversations with representatives fro …  
*Predicted evidence:* Upon receipt of a written request to the undersigned, the undersigned and its Representatives shall promptly return to BDO any and all Information received, printed, downloaded or otherwise copied together with personal notes including written materials and any notes of verbal conversations with representatives from Ba …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Upon receipt of a written request to the undersigned, the undersigned and its Representatives shall promptly return to BDO any and all Information received, printed, downloaded or otherwise copied together with persona …

**doc 16 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* 7. The obligation of non-disclosure shall not extend to:  
*Predicted evidence:* The obligation of non-disclosure shall not extend to: (4) information which the Recipient develops independently of the disclosure as indicated by files existing at the time of initial disclosure.  
*Model said:* { "verdict": "Entailment", "evidence": [ "The obligation of non-disclosure shall not extend to: (4) information which the Recipient develops independently of the disclosure as indicated by files existing at the time of initial disclosure." ] }

**doc 81 · nda-15** — gold `Entailment`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Title or the right to possess Confidential Information as between the parties will remain in the disclosing party.  
*Predicted evidence:* Neither party has any obligation to disclose Confidential Information to the other.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Neither party has any obligation to disclose Confidential Information to the other.", "Both parties understand and acknowledge that no license under any copyrights, trademarks, or maskworks is granted to or conferred u …

**doc 610 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* In the event that a receiving party or its Associates is or becomes legally compelled under applicable law, regulation or securities exchange listing agreement, or by a competent governmental, administrative or regulatory authority or in a proceeding before a court, arbitrator or administrative agency, whether or not a …  
*Predicted evidence:* The receiving party will, and will direct its Associates to, provide the disclosing party with prompt written notice (unless prohibited by law) of such legal compulsion, and shall delay disclosure, if and to the extent permitted or practicable, until the disclaiming party has had an opportunity to seek a protective ord …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The receiving party will, and will direct its Associates to, provide the disclosing party with prompt written notice (unless prohibited by law) of such legal compulsion, and shall delay disclosure, if and to the extent …

**doc 598 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* (i) is or becomes generally available to the public other than as a result of a disclosure, in violation of this agreement, by the Recipient or any of its officers, directors, employees, agents, advisors, members of its advisory committee, lawyers, accountants, auditors or representatives who have been informed of the  …  
*Predicted evidence:* The Recipient shall permit its Representatives access to the Confidential Information only to the extent necessary to allow them to assist the Recipient in the Purpose. The Recipient further agrees that prior to granting such Representatives access to the Confidential Information, the Recipient shall inform such Repres …  
*Model said:* {"verdict": "Entailment", "evidence": ["The Recipient shall permit its Representatives access to the Confidential Information only to the extent necessary to allow them to assist the Recipient in the Purpose. The Recipient further agrees that prior to granting …

**doc 595 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* As used in this letter agreement (this “Agreement”), the term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue a Transaction with you or other parties, the existence and terms of this Lette …  
*Predicted evidence:* As used in this Agreement, the term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue a Transaction with you or other parties, the existence and terms of this Letter Agreement and other info …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "As used in this Agreement, the term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue …

**doc 575 · nda-15** — gold `Entailment`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* All Inventions will be immediately reported to Vapotherm and shall be the exclusive property of Vapotherm.  
*Predicted evidence:* No license or other right is created or granted hereby, except the specific right to receive the Confidential Information and evaluate it as set forth herein, nor shall any license or any other right with respect to the subject matter hereof be created or granted except by written agreement signed by the duly authorize …  
*Model said:* { "verdict": "Entailment", "evidence": [ "No license or other right is created or granted hereby, except the specific right to receive the Confidential Information and evaluate it as set forth herein, nor shall any license or any other right with respect to th …

**doc 69 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* (c) Should it be necessary for a Receiving Party, or anyone to whom it transmits Confidential Information pursuant to this Agreement, to disclose any of the Confidential Information it received from a Disclosing Party in order to comply with applicable law or legal process (including, without limitation, the disclosure …  
*Predicted evidence:* Should it be necessary for a Receiving Party, or anyone to whom it transmits Confidential Information pursuant to this Agreement, to disclose any of the Confidential Information it received from a Disclosing Party in order to comply with applicable law or legal process (including, without limitation, the disclosure req …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Should it be necessary for a Receiving Party, or anyone to whom it transmits Confidential Information pursuant to this Agreement, to disclose any of the Confidential Information it received from a Disclosing Party in o …

**doc 37 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* “Confidential Information” means:  
*Predicted evidence:* “Information” means all information of whatever nature, however conveyed and in whatever form, including in writing, orally, by demonstration, electronically and in a tangible, visual or machine-readable medium (including CD-ROM, magnetic and digital form);  
*Model said:* {"verdict": "Entailment", "evidence": ["“Information” means all information of whatever nature, however conveyed and in whatever form, including in writing, orally, by demonstration, electronically and in a tangible, visual or machine-readable medium (includin …

**doc 406 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* 9. All Proprietary Information furnished by Health Card to Ciufo is considered loaned for use solely in connection with the Project, and shall be returned by Ciufo to Health Card upon request by Health Card.  
*Predicted evidence:* Ciufo (a) must receive and hold the Proprietary Information in trust and in strictest confidence; (b) must protect the Proprietary Information from disclosure and in no event take any action causing, or failing to take reasonable action necessary in order to prevent, any Proprietary Information disclosed to Ciufo to lo …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Ciufo (a) must receive and hold the Proprietary Information in trust and in strictest confidence; (b) must protect the Proprietary Information from disclosure and in no event take any action causing, or failing to take …

**doc 575 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* “Confidential Information” shall, mean all intellectual property, business plans, financial records and strategies, marketing plans, contacts, trade secrets, information, materials, documentation and software, as well as copies of all such materials made thereof as authorized by this Agreement, including written, oral, …  
*Predicted evidence:* “Confidential Information” shall mean all intellectual property, business plans, financial records and strategies, marketing plans, contacts, trade secrets, information, materials, documentation and software, as well as copies of all such materials made thereof as authorized by this Agreement, including written, oral,  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "“Confidential Information” shall mean all intellectual property, business plans, financial records and strategies, marketing plans, contacts, trade secrets, information, materials, documentation and software, as wel …

**doc 73 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* (a) “Confidential Information” shall include all information pertaining to the Assignors and their respective affiliates, employees, salespersons, independent contractors, agents, officers, directors, customers, suppliers, vendors, manufacturers, distributors, dealers, predecessors, successors and assigns disclosed by  …  
*Predicted evidence:* “Confidential Information” shall include all information pertaining to the Assignors and their respective affiliates, employees, salespersons, independent contractors, agents, officers, directors, customers, suppliers, vendors, manufacturers, distributors, dealers, predecessors, successors and assigns disclosed by the  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "“Confidential Information” shall include all information pertaining to the Assignors and their respective affiliates, employees, salespersons, independent contractors, agents, officers, directors, customers, supplie …

**doc 482 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* WHEREAS, the Parties are desirous of exchanging information relating to confidential and more specifically relating, but not limited, to technical data, marketing, business information, and pricing, all for the purpose of exploring future potential business relationships to the mutual benefit of both Parties hereto.  
*Predicted evidence:* 1. "Confidential Information" means all information provided by either Party in writing or visual form which Party desires to protect from disclosure and which is marked with an appropriate stamp or legend designating such material as "Confidential"; and all information provided orally by a Party which that Party desir …  
*Model said:* {"verdict": "Contradiction", "evidence": ["1. \"Confidential Information\" means all information provided by either Party in writing or visual form which Party desires to protect from disclosure and which is marked with an appropriate stamp or legend designati …

**doc 10 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Except as set forth in Section 2.06, the Receiving Party shall not disclose any of the Confidential Information or other facts directly related to the Confidential Information to any Person other than its Representatives who have a need to know such Confidential Information and who have been informed of the confidentia …  
*Predicted evidence:* The Receiving Party agrees that it shall be responsible for any breach of this CNDA by any of its Representatives.  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Receiving Party agrees that it shall be responsible for any breach of this CNDA by any of its Representatives." ] }

**doc 3 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* “Representatives” shall mean directors, employees, professional advisors or anyone involved with the Party in a professional or business capacity.  
*Predicted evidence:* The Receiving Party undertakes to permit access to the Confidential Information only to its Representatives or employees who require access to such Information solely for the fulfilment of the Permitted Purpose, and furnished on a need-to-know basis and only on the condition that such persons are made aware of the term …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Receiving Party undertakes to permit access to the Confidential Information only to its Representatives or employees who require access to such Information solely for the fulfilment of the Permitted Purpose, and fu …

**doc 33 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* "Confidential Information" means any information of a confidential nature relating to the Disclosing Party or a member of its Group disclosed (whether before or after the date of this Agreement and whether in writing, verbally or by any other means and whether directly or indirectly) by the Disclosing Party or by anoth …  
*Predicted evidence:* Confidential Information means any information of a confidential nature relating to the Disclosing Party or a member of its Group disclosed (whether before or after the date of this Agreement and whether in writing, verbally or by any other means and whether directly or indirectly) by the Disclosing Party or by another …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidential Information means any information of a confidential nature relating to the Disclosing Party or a member of its Group disclosed (whether before or after the date of this Agreement and whether in writing, …


### Qwen3-14B (B) — 363 cases, showing 20

**doc 44 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 1. “Confidential Information” shall mean all information, whether communicated in oral, written, electronic or other form, furnished by BOMI or prepared by Attendee during the Testing Period, including, but not limited to, answers and notes made by Attendee during the Testing Period, any test questions and answers, any …  
*Predicted evidence:* “Confidential Information” shall mean all information, whether communicated in oral, written, electronic or other form, furnished by BOMI or prepared by Attendee during the Testing Period,  
*Model said:* <think> Okay, let's see. The user is asking whether the contract says that Confidential Information may include verbally conveyed information. I need to check the contract provided. Looking at the CONTRACT, specifically section 1. The first part defines "Confi …

**doc 482 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 7. Neither Party shall disclose Confidential Information nor assign work hereunder to anyone other than the receiving Party's employees who have agreed in writing to maintain the Confidential Information as confidential and to execute all documents deemed reasonably necessary by the disclosing Party to permit complianc …  
*Predicted evidence:* Neither Party shall disclose Confidential Information nor assign work hereunder to anyone other than the receiving Party's employees who have agreed in writing to maintain the Confidential Information as confidential and to execute all documents deemed reasonably necessary by the disclosing Party to permit compliance w …  
*Model said:* <think> Okay, let's see. The user is asking whether the receiving party can share some confidential information with some of their employees. I need to check the contract to see if that's allowed. Looking through the contract, there's a section that mentions e …

**doc 486 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* d) This Agreement and each additional agreement concluded or written or verbal disclosure made between the Parties, shall be kept confidential and is not to be reproduced, communicated or distributed in any manner whatsoever except on a "need to know" basis to persons directly involved with the closing of any transacti …  
*Predicted evidence:* d) This Agreement and each additional agreement concluded or written or verbal disclosure made between the Parties, shall be kept confidential and is not to be reproduced, communicated or distributed in any manner whatsoever except on a "need to know" basis to persons directly involved with the closing of any transacti …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract states that Confidential Information may include verbally conveyed information. I need to check the contract text for any mention of verbal information being considered confidential. Looking thro …

**doc 54 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* 4. The DISCLOSEE also agrees not to use the confidential information other than for the purpose as provided herein and not to use that information commercially in any way without the DISCLOSER's consent in writing.  
*Predicted evidence:* The DISCLOSEE also agrees not to use the confidential information other than for the purpose as provided herein and not to use that information commercially in any way without the DISCLOSER's consent in writing.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the statement that the Receiving Party (which would be the DISCLOSEE here) shouldn't use Confidential Information for any purpose other than what's stated in the Agreement. Looking at th …

**doc 205 · nda-20** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Gold evidence:* Having acknowledged that it, particularly within a networked environment, the return and/or destruction of electronically stored information may be impossible, extremely or otherwise unduly difficult or costly, the immediately preceding sentence is not intended to, and does not obligate the Receiving Party to make extr …  
*Predicted evidence:* 6.4. Notwithstanding Section 6.3, the Receiving Party may retain a reasonable number of copies of the Confidential Information (and any materials embedding the same) (“Retained Copies”) for the sole purpose of (i) satisfying any legal or regulatory requirements regarding record and data retention that the Receiving Par …  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party can retain some Confidential Information even after returning or destroying it. I need to check the contract for any clauses that mention retention after return or destruction. Looking thr …

**doc 488 · nda-1** — gold `Entailment`, predicted `Entailment`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* 2. Except as set forth in this Section 2, all Confidential Information shall be in tangible form and shall be marked as Confidential or proprietary information of the disclosing party.  
*Predicted evidence:* Except as set forth in this Section 2, all Confidential Information shall be in tangible form and shall be marked as Confidential or proprietary information of the disclosing party. If the Confidential Information is disclosed orally or visually, it shall be identified as such at the time of disclosure and confirmed in …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract states that all Confidential Information must be expressly identified by the Disclosing Party. I need to check the contract clauses to find if there's a provision that requires the disclosing par …

**doc 7 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* (a) 'Confidential Information' shall mean all information received from the other party which the latter has indicated in writing or labelled to be “Confidential”, “Proprietary Information” or with any other comparable legend to similar effect, at the time of disclosure [or if disclosed orally, confirmed in writing by  …  
*Predicted evidence:* (a) 'Confidential Information' shall mean all information received from the other party which the latter has indicated in writing or labelled to be “Confidential”, “Proprietary Information” or with any other comparable legend to similar effect, at the time of disclosure [or if disclosed orally, confirmed in writing by  …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that Confidential Information may include verbally conveyed information. I need to check the contract text for any mention of oral disclosures. Looking through the definitio …

**doc 15 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 3. The receiving party will receive all confidential information (whether recorded in writing or by other means or given orally without record) which is disclosed in connection with this Agreement subject to the following conditions:  
*Predicted evidence:* it will receive all confidential information (whether recorded in writing or by other means or given orally without record) which is disclosed in connection with this Agreement subject to the following conditions:  
*Model said:* <think> Okay, let's see. The user is asking whether the contract says that confidential information can include verbally conveyed information. I need to check the contract text for any mention of oral disclosures. Looking through the contract, in the first par …

**doc 152 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 3. When receiving the Information each party agrees to maintain all such Information in strict confidence and to use at least the same measures to protect the same as that party uses to protect its own confidential information, including but not limited to:  
*Predicted evidence:* restricting the information’s availability to employees, agents or consultants of the receiving party with a need to know, and making such disclosure only after they have agreed to abide by the terms and conditions of this Agreement,  
*Model said:* <think> Okay, let's see. The question is whether the Receiving Party can share some Confidential Information with their employees. I need to check the contract to find relevant clauses. Looking at section 3, it says the receiving party must restrict the inform …

**doc 33 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Each party may disclose the Confidential Information of the other party:  
*Predicted evidence:* 4.1.2 to its employees, professional advisors, authorised representatives, potential underwriters or sub contractors to the extent that disclosure is necessary or desirable for the Purpose;  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party can share Confidential Information with some of their employees. I need to check the contract provided. Looking through the contract, specifically the clauses about disclosure. Clause 4.1. …

**doc 32 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* The receiving party is entitled to use the Confidential Information of the disclosing party solely for the purpose of quoting and manufacturing products for Champion.  
*Predicted evidence:* Confidential Information shall be used by the receiving party only for the purpose set forth in Section 2 of this Agreement and for no other purpose whatsoever.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that the Receiving Party shall not use any Confidential Information for any purpose other than those stated in the Agreement. First, I need to look through the contract to f …

**doc 64 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* The Retained Information shall be held by the Receiving Party, and kept confidential in strict accordance with the terms of this Confidentiality Agreement, or shall be destroyed, unless the Receiving Party is prohibited from destroying it.  
*Predicted evidence:* The Information shall be kept confidential in perpetuity, and in strict accordance with the provisions of this Confidentiality Agreement.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that some obligations of the Agreement may survive termination. First, I need to look through the contract provided. The main document is a Commercial Confi …

**doc 440 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* 4. The obligations under this Agreement shall not apply to information that is at any time:  
*Predicted evidence:* disclosed pursuant to a requirement of a governmental agency or of law, provided, however, that, to the extent viable under the circumstances, the party subject to the disclosure requirement has notified the Disclosing Party in advance of such disclosure  
*Model said:* <think> Okay, let's see. The user is asking whether the contract requires the Receiving Party to notify the Disclosing Party if they have to disclose Confidential Information due to law, regulation, or judicial process. Looking at the contract, specifically pa …

**doc 54 · nda-15** — gold `Entailment`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* 6. No licence, user or similar rights to any patent, patent application or knowhow may be deduced from this agreement, either directly or indirectly.  
*Predicted evidence:* 8. Any and all confidential information, including any and all protected and other rights created and/or produced in connection with the confidential information provided, no matter when or by whom, are the property of Fronius. The DISCLOSEE cannot deduce any rights from knowing the information and knowhow the DISCLOSE …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that the agreement doesn't grant the Receiving Party (which is the DISCLOSEE here) any rights to the Confidential Information. Looking at the contract, I need to check if th …

**doc 56 · nda-20** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Gold evidence:* 3. Each Party acknowledges and agrees:  
*Predicted evidence:* f. To return promptly to the disclosing party, or to destroy any copies of such Confidential Information in written, graphic or other tangible form at that Party’s request and any such destruction shall be certified in writing to the disclosing Party by an authorised officer of the Recipient;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party can retain some confidential information even after returning or destroying it, based on the provided NDA. First, I need to look through the contract terms. The key sectio …

**doc 610 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* Notwithstanding the foregoing, Confidential Information does not include information, technical data or know-how which:  
*Predicted evidence:* Each disclosing party acknowledges that the receiving party or its Associates may currently or in the future be developing information internally, or receiving information from other parties, that is similar to the disclosing party’s Confidential Information. Nothing in this Agreement will prohibit the receiving party  …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract allows the receiving party to independently develop information similar to the confidential information. I need to check the contract terms to find if there's a provision that permits this. Looki …

**doc 3 · nda-11** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.  
*Gold evidence:* The Receiving Parties undertake and agree:  
*Predicted evidence:* f) Not to alter, modify, disassemble, reverse engineer or decompile any Confidential Information;  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party is prohibited from reverse engineering objects that contain the Disclosing Party's Confidential Information. I need to check the contract provided. Looking through the sections, section 3  …

**doc 45 · nda-10** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Gold evidence:* 8. The existence of this Agreement and/or the nature of the business conducted between the parties shall not be disclosed at any time by any party(ies) without the prior written permission of the other party.  
*Predicted evidence:* The existence of this Agreement and/or the nature of the business conducted between the parties shall not be disclosed at any time by any party(ies) without the prior written permission of the other party.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that the Receiving Party shall not disclose the fact that the Agreement was agreed or negotiated. First, I need to look through the contract clauses to find any mention of d …

**doc 27 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* If the Recipient receives a request to disclose all or any part of the Confidential Information under the terms of a valid subpoena or order issued by a court of competent jurisdiction or governmental body, Recipient agrees to  
*Predicted evidence:* If the Recipient receives a request to disclose all or any part of the Confidential Information under the terms of a valid subpoena or order issued by a court of competent jurisdiction or governmental body, Recipient agrees to (i) immediately notify the Discloser of the existence, terms and circumstances surrounding su …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract supports the position that the Receiving Party must notify the Disclosing Party if they are required by law to disclose Confidential Information. First, I need to look through the contract for an …

**doc 590 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* 1.1 “Proprietary Information” means any and all information and material disclosed by Disclosing Party or any of its Representatives to Receiving Party or any of its Representatives in connection with the Proposed Transaction or in the course of the parties’ evaluation and negotiation of the Proposed Transaction, toget …  
*Predicted evidence:* 1.1 "Proprietary Information" means any and all information and material disclosed by Disclosing Party or any of its Representatives to Receiving Party or any of its Representatives in connection with the Proposed Transaction or in the course of the parties’ evaluation and negotiation of the Proposed Transaction, toget …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Confidential Information shall only include technical information." I need to check the contract provided to see if that's the case. First, I'll look  …


### Qwen3-4B (A) — 300 cases, showing 20

**doc 440 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 4. The obligations under this Agreement shall not apply to information that is at any time:  
*Predicted evidence:* The obligations under this Agreement shall not apply to information that is at any time: (c) rightfully received from a third party without restriction on disclosure and without breach of this Agreement;  
*Model said:* {"verdict": "Entailment", "evidence": ["The obligations under this Agreement shall not apply to information that is at any time: (c) rightfully received from a third party without restriction on disclosure and without breach of this Agreement;"]}

**doc 16 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* 2. Recipient acknowledges the sensitive and secret nature of the Confidential Information it will have access to during the term of this Agreement and agrees that it will treat such Confidential Information as strictly confidential and shall exercise the same degree of care in the protection of the Confidential Informa …  
*Predicted evidence:* 10. The representations and obligations of the Recipient relative to the Confidential Information shall survive the termination of this Confidentiality Agreement for a period up to three (3) years from the date of the completion or other termination of the Project.  
*Model said:* { "verdict": "Entailment", "evidence": [ "10. The representations and obligations of the Recipient relative to the Confidential Information shall survive the termination of this Confidentiality Agreement for a period up to three (3) years from the date of the  …

**doc 507 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* Notwithstanding the foregoing, the above restrictions shall not apply to the following:  
*Predicted evidence:* (ii) Information that is disclosed to Holdings by a third party that did not disclose it in violation of a duty of confidentiality;  
*Model said:* {"verdict": "Entailment", "evidence": ["(ii) Information that is disclosed to Holdings by a third party that did not disclose it in violation of a duty of confidentiality;"]}

**doc 381 · nda-18** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not solicit some of Disclosing Party's representatives.  
*Gold evidence:* (a) For the period beginning on the Effective Date and ending on the * anniversary thereof (the "Restriction Period"), neither Selling Parties nor any subsidiary, sister entity, or parent will, directly or indirectly, either as a partner or owner or in any other capacity:  
*Predicted evidence:* Selling Parties represent to the Company and Meritage, and Selling Parties acknowledge, that: (ii) enforcement of the restrictions set forth in this Section 1 would not be unduly burdensome to Selling Parties;  
*Model said:* { "verdict": "Entailment", "evidence": [ "Selling Parties represent to the Company and Meritage, and Selling Parties acknowledge, that: (ii) enforcement of the restrictions set forth in this Section 1 would not be unduly burdensome to Selling Parties;" ] }

**doc 70 · nda-7** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* I hereby agree as follows:  
*Predicted evidence:* I will not disclose any PHI to any individual or third party, except as specifically authorized by FAU policies and procedures, and upon receiving a written authorization from the patient (unless otherwise required by applicable law), and then only on a need-to-know basis.  
*Model said:* {"verdict": "Contradiction", "evidence": ["I will not disclose any PHI to any individual or third party, except as specifically authorized by FAU policies and procedures, and upon receiving a written authorization from the patient (unless otherwise required by …

**doc 19 · nda-7** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* a) Confidential Information shall be kept in the strictest confidence without limitation of time, and shall not be disclosed to any Third Party;  
*Predicted evidence:* Confidant shall restrict access to Confidential Information only to its employees with a need to know to carry out the Permitted Use, and prior to disclosing same, each such employee shall be made aware of the terms and conditions of this Agreement; and Confidant shall cause all of its applicable employees to observe t …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Confidant shall restrict access to Confidential Information only to its employees with a need to know to carry out the Permitted Use, and prior to disclosing same, each such employee shall be made aware of the terms a …

**doc 488 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* 8. This Agreement shall govern all communications between the parties that are made during the period from the effective date of this Agreement to the date on which either party receives from the other written notice that subsequent communications shall not be so governed, provided, however, that each party's obligatio …  
*Predicted evidence:* Neither party shall communicate any information to the other in violation of the proprietary rights of any third party. Neither party acquires any licenses under any intellectual property rights of the other party under this Agreement.  
*Model said:* {"verdict": "Entailment", "evidence": ["Neither party shall communicate any information to the other in violation of the proprietary rights of any third party. Neither party acquires any licenses under any intellectual property rights of the other party under  …

**doc 590 · nda-20** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Gold evidence:* In that case, or if the  
*Predicted evidence:* However, Receiving Party may retain in the office of its legal counsel, one copy of Proprietary Information for record purposes only.  
*Model said:* {"verdict": "Entailment", "evidence": ["However, Receiving Party may retain in the office of its legal counsel, one copy of Proprietary Information for record purposes only."]}

**doc 547 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 2011. Confidential Information, regardless of the form in which it appears, may include, without limitation, information about customers, business and marketing plans, financial data, expertise of employees, forecasts and strategies that is related to the business of the Disclosing Party and is of the type of informati …  
*Predicted evidence:* Confidential Information, regardless of the form in which it appears, may include, without limitation, information about customers, business and marketing plans, financial data, expertise of employees, forecasts and strategies that is related to the business of the Disclosing Party and is of the type of information tha …  
*Model said:* {"verdict": "Entailment", "evidence": ["Confidential Information, regardless of the form in which it appears, may include, without limitation, information about customers, business and marketing plans, financial data, expertise of employees, forecasts and stra …

**doc 488 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 4. There shall be no liability for disclosure or use of Confidential Information which is  
*Predicted evidence:* There shall be no liability for disclosure or use of Confidential Information which is (b) rightfully received from a third party without any obligation of confidentiality,  
*Model said:* {"verdict": "Entailment", "evidence": ["There shall be no liability for disclosure or use of Confidential Information which is (b) rightfully received from a third party without any obligation of confidentiality,"]}

**doc 595 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* As used in this letter agreement (this “Agreement”), the term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue a Transaction with you or other parties, the existence and terms of this Lette …  
*Predicted evidence:* The term "Evaluation Material" does not include any information which (i) at the time of disclosure or thereafter is generally known by the public (other than as a result of its disclosure by you or your Representatives) or (ii) was or becomes available to you on a non-confidential basis from a person not otherwise bou …  
*Model said:* {"verdict": "Entailment", "evidence": ["The term \"Evaluation Material\" does not include any information which (i) at the time of disclosure or thereafter is generally known by the public (other than as a result of its disclosure by you or your Representative …

**doc 27 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Confidential Information may be used only to the extent reasonably required for the purposes of evaluating, producing or carrying out the Transaction; provided however, that Recipients may only disclose the Confidential Information to those of its contractors, vendors, directors, officers, advisors, agents and employee …  
*Predicted evidence:* (none)  
*Model said:* {"verdict": "Entailment", "evidence": ["Recipients may only disclose the Confidential Information to those of its contractors, vendors, directors, officers, advisors, agents and employees ("Representatives") who need to know such Confidential Information for t …

**doc 27 · nda-1** — gold `Entailment`, predicted `Entailment`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* Each Party receiving Confidential Information (as defined below) (each, a "Recipient," and together, the "Recipients"), is receiving confidential disclosure of certain confidential or proprietary technical, business, operational, financial and/or commercial information, trade secret information and/or any information o …  
*Predicted evidence:* Specifically, with respect to the Producer, the term “Confidential Information” includes confidential and proprietary information of Producer disclosed in connection with the Transaction (during each phase including pre-production, conference(s), pre-bid, bid, production, shoot, wrap and post-production) and regardless …  
*Model said:* {"verdict": "Entailment", "evidence": ["Specifically, with respect to the Producer, the term “Confidential Information” includes confidential and proprietary information of Producer disclosed in connection with the Transaction (during each phase including pre- …

**doc 81 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is  
*Predicted evidence:* The disclosing party shall make reasonable efforts to mark its confidential information in tangible form with any of the aforementioned legends prior to disclosure. However, the disclosing party’s information in tangible form that does not bear any of these legends, and discussions relating to that information, shall n …  
*Model said:* {"verdict": "Entailment", "evidence": ["The disclosing party shall make reasonable efforts to mark its confidential information in tangible form with any of the aforementioned legends prior to disclosure. However, the disclosing party’s information in tangible …

**doc 63 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* 2. Upon receipt of any Confidential Information, Receiving Party shall:  
*Predicted evidence:* Receiving Party shall use, and require its Representatives to use, the Confidential Information only for the purpose expressly stated herein; provided that nothing in this Agreement is intended to limit or prevent either Party from continuing to operate its business in the ordinary course, including conducting business …  
*Model said:* {"verdict": "Entailment", "evidence": ["Receiving Party shall use, and require its Representatives to use, the Confidential Information only for the purpose expressly stated herein; provided that nothing in this Agreement is intended to limit or prevent either …

**doc 63 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* 14. All obligations of Receiving Party with respect to any Confidential Information received from Disclosing Party under the terms of this Agreement shall survive and continue for five (5) years after the date of the last disclosure of Confidential Information by Disclosing Party.  
*Predicted evidence:* All obligations of Receiving Party with respect to any Confidential Information received from Disclosing Party under the terms of this Agreement shall survive and continue for five (5) years after the date of the last disclosure of Confidential Information by Disclosing Party.  
*Model said:* {"verdict": "Entailment", "evidence": ["All obligations of Receiving Party with respect to any Confidential Information received from Disclosing Party under the terms of this Agreement shall survive and continue for five (5) years after the date of the last di …

**doc 56 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* g. The obligations of each of the Parties with respect to Confidential Information shall extend for a period of three (3) years following the date of disclosure of that Confidential Information; provided, however, that confidentiality obligations herein shall survive for a period of three (3) years from the date of ter …  
*Predicted evidence:* g. The obligations of each of the Parties with respect to Confidential Information shall extend for a period of three (3) years following the date of disclosure of that Confidential Information; provided, however, that confidentiality obligations herein shall survive for a period of three (3) years from the date of ter …  
*Model said:* {"verdict": "Entailment", "evidence": ["g. The obligations of each of the Parties with respect to Confidential Information shall extend for a period of three (3) years following the date of disclosure of that Confidential Information; provided, however, that c …

**doc 56 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* 3. Each Party acknowledges and agrees:  
*Predicted evidence:* To use such Confidential Information only for purposes of work, services or analysis related to the Matter and for other purposes only upon such terms as may be agreed upon between the Parties in writing;  
*Model said:* {"verdict": "Entailment", "evidence": ["To use such Confidential Information only for purposes of work, services or analysis related to the Matter and for other purposes only upon such terms as may be agreed upon between the Parties in writing;"]}

**doc 29 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* CI does not include information that  
*Predicted evidence:* Each party agrees that Recipient (A) may independently develop and acquire software, hardware, and other technology for its own products, services, and processes, and that having access to Discloser’s CI does not prohibit Recipient from freely developing or acquiring similar products without obligation to Discloser; an …  
*Model said:* {"verdict": "Entailment", "evidence": ["Each party agrees that Recipient (A) may independently develop and acquire software, hardware, and other technology for its own products, services, and processes, and that having access to Discloser’s CI does not prohibi …

**doc 63 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 1. Subject to the limitations set forth herein, “Confidential Information” means any and all confidential, proprietary or trade secret information of Disclosing Party (whether oral, written, electronic or otherwise), including but not limited to marketing and sales data, customer lists, customer requirements, price lis …  
*Predicted evidence:* 1. Subject to the limitations set forth herein, "Confidential Information" means any and all confidential, proprietary or trade secret information of Disclosing Party (whether oral, written, electronic or otherwise), including but not limited to marketing and sales data, customer lists, customer requirements, price lis …  
*Model said:* {"verdict": "Entailment", "evidence": ["1. Subject to the limitations set forth herein, \"Confidential Information\" means any and all confidential, proprietary or trade secret information of Disclosing Party (whether oral, written, electronic or otherwise), i …


### Qwen3-4B (B) — 341 cases, showing 20

**doc 28 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 4. The restrictions of this Agreement on use and disclosure of Confidential Information shall not apply to information, which is:  
*Predicted evidence:* 4. The restrictions of this Agreement on use and disclosure of Confidential Information shall not apply to information, which is: (d) rightfully obtained by the Recipient from third parties authorized to make such disclosure without restriction;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to acquire information similar to Confidential Information from a third party. The answer options are Entailment, Contradiction, or NotMentioned. First …

**doc 605 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* The term Blueknight Non-public Information shall not include, however, information which  
*Predicted evidence:* Section 2(b)(iii) of the contract states: 'The term Blueknight Non-public Information shall not include, however, information which (iii) is developed independently by Receiving Party without use of Blueknight Non-public Information and without violating any of the provisions of this Agreement.'  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party may independently develop information similar to Confidential Information. The answer needs to be either Entailment, Contradiction, …

**doc 582 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 1. You hereby agree that the Confidential Information will be used solely for the purpose of evaluating and implementing the Transaction and that such information will be kept confidential by you; provided that any such information may be disclosed to your officers, directors, employees, affiliates, accountants, attorn …  
*Predicted evidence:* You hereby agree that the Confidential Information will be used solely for the purpose of evaluating and implementing the Transaction and that such information will be kept confidential by you; provided that any such information may be disclosed to your officers, directors, employees, affiliates, accountants, attorneys …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to share some Confidential Information with some of their employees. The answer needs to be either Entailment, Contradiction, or NotMentioned, based on …

**doc 16 · nda-17** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may create a copy of some Confidential Information in some circumstances.  
*Gold evidence:* 4. Recipient shall not make copies of the Confidential Information except for those copies required for use by Authorized Employees in the performance of the Agreement.  
*Predicted evidence:* Recipient shall not make copies of the Confidential Information except for those copies required for use by Authorized Employees in the performance of the Agreement. Each copy, including its storage media, shall be marked Confidential, and also include all copyright, trademark and other proprietary notices which appear …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party (Recipient) may create a copy of some Confidential Information in some circumstances. The answer needs to be either Entailment, Con …

**doc 478 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* 4. The Receiving Party shall use Proprietary Information received from the Disclosing Party solely for the purposes set forth in Paragraph 1.  
*Predicted evidence:* The confidentiality obligations under this Agreement will survive termination of this Agreement for a period of three (3) years.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that some obligations of the Agreement may survive termination. The answer needs to be either Entailment, Contradiction, or NotMentioned, with evidence from …

**doc 488 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* 4. There shall be no liability for disclosure or use of Confidential Information which is  
*Predicted evidence:* provided that the party so disclosing has provided the other party with a reasonable opportunity to seek protective legal treatment for such Confidential Information  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party must notify the Disclosing Party if they're required by law, regulation, or judicial process to disclose Confidential Information.  …

**doc 33 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* In consideration of each party supplying its Confidential Information to the other party and subject to the exceptions described in Clause 5, the parties undertake as follows:  
*Predicted evidence:* 2.2 not to use the other party's Confidential Information except for the Purpose  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party shall not use any Confidential Information for any purpose other than the ones stated in the Agreement. First, I need to look throu …

**doc 35 · nda-18** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not solicit some of Disclosing Party's representatives.  
*Gold evidence:* The Receiving Party agrees that, save as contemplated in any definitive transaction agreement concluded between the Parties for purposes of the Transaction, it will not, for a period of 24 months after the Signature Date, directly or indirectly, and on its own behalf or on behalf of any other person, encourage, entice, …  
*Predicted evidence:* 7.1 who is, as at the Signature Date or during the course of the Parties discussions relating to the Purpose, in the employ of the Disclosing Party or its Affiliates; and  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party shall not solicit some of the Disclosing Party's representatives. The answer needs to be either Entailment, Contradiction, or NotMe …

**doc 75 · nda-20** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Gold evidence:* If the Investor elects not to proceed with the Development, or if the undersigned is informed that the Property is no longer available for purchase or if the Investor is otherwise requested by the Owner to return the Confidential Information, the Investor will promptly return all Confidential Information (and all repro …  
*Predicted evidence:* Investor will promptly return all Confidential Information (and all reproductions or destroy electronic copies thereof if any);  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which here is the Investor) to retain some Confidential Information even after returning or destroying it. The answer needs to be based on the provide …

**doc 69 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* (f) “Representatives” means all Persons acting on behalf of a specified Party or its Affiliates, including without limitation, its directors, officers, lenders, employees, agents, representatives, financial advisors, attorneys, accountants, consultants, reserve and valuation and other experts.  
*Predicted evidence:* Each Party shall be responsible for any actions by its Affiliates and/or Representatives which are not in accordance with the agreements herein regarding the conduct of its Affiliates and/or Representatives. Visitors shall not take photographs or otherwise record the activities at the other party’s facilities.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may share some Confidential Information with some of their employees. The contract in question is a Mutual Confidentiality and Non-Disclosure Agreement. First, I need to l …

**doc 56 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* g. The obligations of each of the Parties with respect to Confidential Information shall extend for a period of three (3) years following the date of disclosure of that Confidential Information; provided, however, that confidentiality obligations herein shall survive for a period of three (3) years from the date of ter …  
*Predicted evidence:* The obligations of each of the Parties with respect to Confidential Information shall extend for a period of three (3) years following the date of disclosure of that Confidential Information; provided, however, that confidentiality obligations herein shall survive for a period of three (3) years from the date of termin …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that some obligations of the Agreement may survive termination. The answer needs to be either Entailment, Contradiction, or NotMentioned, with evidence from …

**doc 81 · nda-15** — gold `Entailment`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Title or the right to possess Confidential Information as between the parties will remain in the disclosing party.  
*Predicted evidence:* 5. Title. Title or the right to possess Confidential Information as between the parties will remain in the disclosing party.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement does not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment, Contradiction, or Not …

**doc 205 · nda-20** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Gold evidence:* Having acknowledged that it, particularly within a networked environment, the return and/or destruction of electronically stored information may be impossible, extremely or otherwise unduly difficult or costly, the immediately preceding sentence is not intended to, and does not obligate the Receiving Party to make extr …  
*Predicted evidence:* 6.4. Notwithstanding anything else in this Agreement, the Receiving Party’s obligations of confidentiality and restricted use hereunder shall continue in force during the term of this Agreement and shall survive for three (3) years following the expiration or termination of this Agreement or, in the case of trade secre …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to retain some Confidential Information even after returning or destroying it. The answer options are Entailment, Contradiction, or NotMentioned. First …

**doc 19 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* b) Confidant shall restrict access to Confidential Information only to its employees with a need to know to carry out the Permitted Use, and prior to disclosing same, each such employee shall be made aware of the terms and conditions of this Agreement; and  
*Predicted evidence:* Confidant shall restrict access to Confidential Information only to its employees with a need to know to carry out the Permitted Use, and prior to disclosing same, each such employee shall be made aware of the terms and conditions of this Agreement;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which here is the Confidant) to share some Confidential Information with some of their employees. The answer needs to be either Entailment, Contradict …

**doc 205 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* 1.1. “Confidential Information” shall mean information that is revealed by or obtained through a party (whether in writing, orally or by any other means) (a “Disclosing Party”) to the other (a “Receiving Party”) including, without limitation,  
*Predicted evidence:* 1.1. "Confidential Information" shall mean information that is revealed by or obtained through a party (whether in writing, orally or by any other means) (a "Disclosing Party") to the other (a "Receiving Party") including, without limitation, (i) all forms and types of financial, business, scientific, technical, econom …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Confidential Information shall only include technical information." The answer needs to be either Entailment, Contradiction, or NotMentioned, with evi …

**doc 37 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* The Receiving Party and each Receiving Party Authorised Person shall be entitled to disclose Confidential Information to the extent that it is required to do so by applicable law or by order of a court or other public body that has jurisdiction over the Receiving Party or Receiving Party Authorised Person, provided tha …  
*Predicted evidence:* 3.3.1 notify the Disclosing Party in writing of the proposed disclosure as soon as possible (and if possible before the court or other public body orders the disclosure of the Confidential Information)  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party must notify the Disclosing Party if they are required by law, regulation, or judicial process to disclose Confidential Information. …

**doc 405 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* You and your representatives will keep the Evaluation Materials completely confidential and will not disclose the Evaluation Materials, in any manner, in whole or in part; provided, however, that  
*Predicted evidence:* You and your representatives will not use any of the Evaluation Materials for any purpose other than the exclusive purpose of evaluating a Possible Transaction. You and your representatives will keep the Evaluation Materials completely confidential and will not disclose the Evaluation Materials, in any manner, in whole …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which is United Dominion Industries in this case) to share some Confidential Information with some of their employees. The answer needs to be determin …

**doc 19 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RFP and/or the Permitted Use, which comes into the possession or knowledge of the Confidant, including …  
*Predicted evidence:* "Confidential Information" means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RFP and/or the Permitted Use, which comes into the possession or knowledge of the Confidant, including, w …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that Confidential Information may include verbally conveyed information. The answer needs to be either Entailment, Contradiction, or NotMentioned, with evid …

**doc 56 · nda-7** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* 2. The Recipient shall be entitled to disclose the Confidential Information to its employees, contractors and advisers and Group Parties, provided that the Recipient will ensure that any such employee, contractor or adviser or Group Party has a legal obligation to keep the Confidential Information confidential which, i …  
*Predicted evidence:* The Recipient shall be entitled to disclose the Confidential Information to its employees, contractors and advisers and Group Parties, provided that the Recipient will ensure that any such employee, contractor or adviser or Group Party has a legal obligation to keep the Confidential Information confidential which, in t …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to share Confidential Information with third parties like consultants, agents, and professional advisors. First, I need to look through the contract te …

**doc 12 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 1.2 BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not having reasonable need for access to such information, unless CLIENT expressly consents to such disclo …  
*Predicted evidence:* BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not having reasonable need for access to such information, unless CLIENT expressly consents to such disclosure …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which in this case is BROOKS) to share some Confidential Information with some of their employees. The answer needs to be determined based on the cont …


### Qwen3-8B (A) — 376 cases, showing 20

**doc 20 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* 2.1  
*Predicted evidence:* 2.1 For the purpose of this Agreement, "Confidential Information" means any information received by a Party (the "Receiving Party") regarding the other Party’s (the "Disclosing Party") business, research, products and/or services, such as all information and technology, including without limitation, research, invention …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2.1 For the purpose of this Agreement, \"Confidential Information\" means any information received by a Party (the \"Receiving Party\") regarding the other Party’s (the \"Disclosing Party\") business, research, prod …

**doc 547 · nda-1** — gold `Contradiction`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* 2011. Confidential Information, regardless of the form in which it appears, may include, without limitation, information about customers, business and marketing plans, financial data, expertise of employees, forecasts and strategies that is related to the business of the Disclosing Party and is of the type of informati …  
*Predicted evidence:* The term "Confidential Information" means all confidential and proprietary information belonging to one party (the "Disclosing Party") that is revealed to the other party (the "Receiving Party") during the Disclosure Period  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The term \"Confidential Information\" means all confidential and proprietary information belonging to one party (the \"Disclosing Party\") that is revealed to the other party (the \"Receiving Party\") during the Dis …

**doc 605 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* (b) “Blueknight Non-public Information” shall mean information pertaining to Blueknight, the General Partner and/or their respective Affiliates that has not been Publicly Disclosed and that Blueknight or the General Partner discloses to Receiving Party, including, without limitation, the existence of the Refinancing Me …  
*Predicted evidence:* Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made available to Receiving Party  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Receiving Party shall (a) keep and maintain any Material Blueknight Non-public Information in strict confidence and (b) not disclose to any other person the fact that Blueknight Non-public Information has been made  …

**doc 595 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* As used in this letter agreement (this “Agreement”), the term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue a Transaction with you or other parties, the existence and terms of this Lette …  
*Predicted evidence:* The term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue a Transaction with you or other parties, the existence and terms of this Letter Agreement and other information furnished to you in …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue a Transaction with you or  …

**doc 440 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* 2. Each party agrees to hold the other's Confidential Information in confidence and to use it solely for the purposes contemplated herein and agrees further that the Confidential Information shall not be used for any other purpose nor disclosed to any third party without the prior written consent of the other party.  
*Predicted evidence:* Each party agrees to hold the other's Confidential Information in confidence and to use it solely for the purposes contemplated herein and agrees further that the Confidential Information shall not be used for any other purpose nor disclosed to any third party without the prior written consent of the other party.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Each party agrees to hold the other's Confidential Information in confidence and to use it solely for the purposes contemplated herein and agrees further that the Confidential Information shall not be used for any othe …

**doc 563 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 1. “Confidential Information” means, for each party, whether in oral, written or in electronic form:  
*Predicted evidence:* The Recipient will keep, and will cause its Representatives to keep, the Confidential Information of the Disclosing Party confidential. The Recipient will not disclose the Confidential Information or use or permit the Confidential Information to be disclosed or used, to or by any person except Recipient’s Representativ …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Recipient will keep, and will cause its Representatives to keep, the Confidential Information of the Disclosing Party confidential. The Recipient will not disclose the Confidential Information or use or permit the  …

**doc 586 · nda-7** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* (iii) the term "Representatives" means the directors, officers, employees, agents or advisors (including, without limitation, attorneys, accountants, investment bankers, financial advisors and other consultants and advisors engaged in connection with the review and evaluation of the Transaction) of the specified Party; …  
*Predicted evidence:* The Receiving Party shall inform such Representative of the confidentiality of the Evaluation Material, and shall advise such Representative that, by accepting possession of or access to such information, such Representative is agreeing to be bound by this Agreement.  
*Model said:* { "verdict": "Entailment", "evidence": [ "The Receiving Party shall inform such Representative of the confidentiality of the Evaluation Material, and shall advise such Representative that, by accepting possession of or access to such information, such Represen …

**doc 63 · nda-10** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Gold evidence:* 10. Each Party hereto agrees, and shall cause its Representatives, not to announce or disclose to any third person its participation in discussions with the other party concerning the matters contained in this Agreement without the advance written consent of the other Party.  
*Predicted evidence:* Each Party hereto agrees, and shall cause its Representatives, not to announce or disclose to any third person its participation in discussions with the other party concerning the matters contained in this Agreement without the advance written consent of the other Party.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Each Party hereto agrees, and shall cause its Representatives, not to announce or disclose to any third person its participation in discussions with the other party concerning the matters contained in this Agreement wi …

**doc 81 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* The receiving party will not be liable for the disclosure of any Confidential Information which is:  
*Predicted evidence:* The receiving party will not be liable for the disclosure of any Confidential Information which is... required to be disclosed in accordance with applicable laws, regulations, court, judicial or other government order, provided that the receiving party shall give the disclosing party reasonable notice prior to such dis …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The receiving party will not be liable for the disclosure of any Confidential Information which is... required to be disclosed in accordance with applicable laws, regulations, court, judicial or other government order, …

**doc 73 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* (a) “Confidential Information” shall include all information pertaining to the Assignors and their respective affiliates, employees, salespersons, independent contractors, agents, officers, directors, customers, suppliers, vendors, manufacturers, distributors, dealers, predecessors, successors and assigns disclosed by  …  
*Predicted evidence:* Confidential Information shall include all information pertaining to the Assignors and their respective affiliates, employees, salespersons, independent contractors, agents, officers, directors, customers, suppliers, vendors, manufacturers, distributors, dealers, predecessors, successors and assigns disclosed by the Di …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Confidential Information shall include all information pertaining to the Assignors and their respective affiliates, employees, salespersons, independent contractors, agents, officers, directors, customers, suppliers,  …

**doc 39 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* By signing this agreement the Bidder is stating that it will keep all Confidential Information (as defined in paragraph 4) in strict confidence and only use the Confidential Information on the Project identified herein.  
*Predicted evidence:* The purpose of this agreement is to allow the Bidder access to Confidential Information as defined herein detailing the design of the Project for the sole purpose of compiling and submitting a bid in response to the County’s Invitation for Bid to construct the Project.  
*Model said:* { "verdict": "Entailment", "evidence": [ "The purpose of this agreement is to allow the Bidder access to Confidential Information as defined herein detailing the design of the Project for the sole purpose of compiling and submitting a bid in response to the Co …

**doc 56 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* 3. Each Party acknowledges and agrees:  
*Predicted evidence:* To use such Confidential Information only for purposes of work, services or analysis related to the Matter and for other purposes only upon such terms as may be agreed upon between the Parties in writing;  
*Model said:* { "verdict": "Entailment", "evidence": [ "To use such Confidential Information only for purposes of work, services or analysis related to the Matter and for other purposes only upon such terms as may be agreed upon between the Parties in writing;" ] }

**doc 19 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* c) “Confidential Information” means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RFP and/or the Permitted Use, which comes into the possession or knowledge of the Confidant, including …  
*Predicted evidence:* Confidential Information means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RFP and/or the Permitted Use, which comes into the possession or knowledge of the Confidant, including, wit …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidential Information means any and all information, regardless of form, format or medium (including without limitation visual or oral information), of, related to, concerning, or resulting from, the City, the RF …

**doc 488 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* 1. "Confidential Information" is that confidential, proprietary, and trade secret information being disclosed by the disclosing party pursuant to this Agreement.  
*Predicted evidence:* Confidential Information is that confidential, proprietary, and trade secret information being disclosed by the disclosing party pursuant to this Agreement.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Confidential Information is that confidential, proprietary, and trade secret information being disclosed by the disclosing party pursuant to this Agreement." ] }

**doc 398 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* (c) With respect to any trade secrets included in the Confidential Information, Seller also agrees not to use or disclose any of such trade secrets at any time until such trade secrets become generally available to the public by independent discovery or development and publication through no fault of Seller.  
*Predicted evidence:* 3.07 BINDING EFFECT. This Agreement shall be binding upon and inure to the benefit of the parties and their respective heirs, executors, administrators, successors, and permitted assigns.  
*Model said:* { "verdict": "Entailment", "evidence": [ "3.07 BINDING EFFECT. This Agreement shall be binding upon and inure to the benefit of the parties and their respective heirs, executors, administrators, successors, and permitted assigns." ] }

**doc 435 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* 4. There shall be no liability for disclosure or use of Confidential Information which is  
*Predicted evidence:* There shall be no liability for disclosure or use of Confidential Information which is (a) in the public domain through no fault of the receiving party (b) rightfully received from a third party without any obligation of confidentiality, (c) rightfully known to the receiving party without any limitation on use or discl …  
*Model said:* { "verdict": "Entailment", "evidence": [ "There shall be no liability for disclosure or use of Confidential Information which is (a) in the public domain through no fault of the receiving party (b) rightfully received from a third party without any obligation  …

**doc 33 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* Without prejudice to the generality of Clause 8.2, each party may retain:  
*Predicted evidence:* 8.2 Each party undertakes within 14 days of receipt of a written request of the other party: ... to destroy by shredding or incineration all documents and other material in its possession, custody or control which reflect or have been generated from any part of the other party's Confidential Information and to certify  …  
*Model said:* { "verdict": "Entailment", "evidence": [ "8.2 Each party undertakes within 14 days of receipt of a written request of the other party: ... to destroy by shredding or incineration all documents and other material in its possession, custody or control which refl …

**doc 563 · nda-1** — gold `Contradiction`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* 1. “Confidential Information” means, for each party, whether in oral, written or in electronic form:  
*Predicted evidence:* All right, title and interest in and to a Disclosing Party’s Confidential Information will remain the exclusive property of the Disclosing Party and the Confidential Information will be held in trust and confidence by the Recipient and its Representatives. No interest, license or any right respecting the Confidential I …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "All right, title and interest in and to a Disclosing Party’s Confidential Information will remain the exclusive property of the Disclosing Party and the Confidential Information will be held in trust and confidence  …

**doc 35 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* Should the Receiving Party or any of its employees, officers, directors, representatives, advisers or Affiliates be obliged by any law, or a valid order of any court of competent jurisdiction or government authority acting with the powers granted to it in law, to disclose any Confidential Information of the Disclosing  …  
*Predicted evidence:* 4.5 Should the Receiving Party or any of its employees, officers, directors, representatives, advisers or Affiliates be obliged by any law, or a valid order of any court of competent jurisdiction or government authority acting with the powers granted to it in law, to disclose any Confidential Information of the Disclos …  
*Model said:* { "verdict": "Entailment", "evidence": [ "4.5 Should the Receiving Party or any of its employees, officers, directors, representatives, advisers or Affiliates be obliged by any law, or a valid order of any court of competent jurisdiction or government authorit …

**doc 488 · nda-11** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.  
*Gold evidence:* 6. Each of the parties agrees that it will not modify, reverse engineer, decompile, create other works from, or disassemble any software programs contained in the Confidential Information of the other party unless otherwise specified in writing by the disclosing party.  
*Predicted evidence:* Each of the parties agrees that it will not modify, reverse engineer, decompile, create other works from, or disassemble any software programs contained in the Confidential Information of the other party unless otherwise specified in writing by the disclosing party.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Each of the parties agrees that it will not modify, reverse engineer, decompile, create other works from, or disassemble any software programs contained in the Confidential Information of the other party unless otherwi …


### Qwen3-8B (B) — 319 cases, showing 20

**doc 3 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* In respect of Confidential Information disclosed in documentary form, model or any other tangible form, this shall be marked by the Disclosing Party as confidential or otherwise designated to show expressly or by necessary implication that it is imparted in confidence.  
*Predicted evidence:* Any Confidential Information disclosed by the Disclosing Party or its Representatives orally or visually, shall be identified as such prior to disclosure and summarized in a writing provided by the Disclosing Party to the Receiving Party within thirty (30) days of the date of initial disclosure.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Confidential Information may include verbally conveyed information." First, I need to look through the contract text provided. The key section here i …

**doc 20 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 2.2 Confidential Information shall not include information that:  
*Predicted evidence:* 2.2 Confidential Information shall not include information that: iii. was lawfully obtained by the Receiving Party from a third party;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may acquire information similar to Confidential Information from a third party. The contract is a Mutual Non-Disclosure Agreement, so I need to check the sections related  …

**doc 14 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* 3. If the Stockholder is required (by interrogatory, subpoena, civil investigatory demand, or any similar process relating to any legal proceeding, investigation, hearing, or otherwise) to disclose in any manner any Confidential Material, the Stockholder will provide the Company with prompt notice (i.e., within 2 busin …  
*Predicted evidence:* 3. If the Stockholder is required (by interrogatory, subpoena, civil investigatory demand, or any similar process relating to any legal proceeding, investigation, hearing, or otherwise) to disclose in any manner any Confidential Material, the Stockholder will provide the Company with prompt notice (i.e., within 2 busin …  
*Model said:* <think> Okay, let's tackle this query. The user wants to know if the contract supports, contradicts, or doesn't mention the statement about the Receiving Party notifying the Disclosing Party when required by law to disclose Confidential Information. First, I n …

**doc 7 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* (a) 'Confidential Information' shall mean all information received from the other party which the latter has indicated in writing or labelled to be “Confidential”, “Proprietary Information” or with any other comparable legend to similar effect, at the time of disclosure [or if disclosed orally, confirmed in writing by  …  
*Predicted evidence:* (B) The parties wish to define their rights with respect to the said information and to protect the confidentiality thereof and proprietary features contained therein.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Confidential Information may include verbally conveyed information." I need to check the contract to see if it mentions verbal information as part of …

**doc 61 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 3. Company shall not  
*Predicted evidence:* 3. Company shall not (i) use any portion of the Confidential Information for any purpose except to evaluate the RFP... (d) is received from a third party free to disclose it to Company;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which would be the Company in this case) to acquire information similar to Confidential Information from a third party. First, I need to look at the r …

**doc 406 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* b. Proprietary Information does not include any information which  
*Predicted evidence:* b. Proprietary Information does not include any information which ... (iv) has been independently developed by Ciufo without use, directly or indirectly, of the Proprietary Information;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which would be Ciufo in this case) to independently develop information similar to Confidential Information. First, I need to recall the relevant part …

**doc 3 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* “Representatives” shall mean directors, employees, professional advisors or anyone involved with the Party in a professional or business capacity.  
*Predicted evidence:* 4. Disclosure to Representatives The Parties shall use at least the same degree of care as the Receiving Party normally exercises to protect its own information of like character and importance, but in no event less than reasonable care. The Receiving Party undertakes to permit access to the Confidential Information on …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may share some Confidential Information with some of their employees, based on the provided contract. First, I need to recall the relevant sections of the contract. The ke …

**doc 64 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* E. COMPELLED DISCLOSURE: In the event that the Receiving Party or anyone to whom the Receiving Party transmits the Information pursuant to this Confidentiality Agreement, becomes legally compelled to disclose any of the Information, the Receiving Party shall provide the Delivering Party with prompt Notice so that the D …  
*Predicted evidence:* In the event that the Receiving Party or anyone to whom the Receiving Party transmits the Information pursuant to this Confidentiality Agreement, becomes legally compelled to disclose any of the Information, the Receiving Party shall provide the Delivering Party with prompt Notice so that the Delivering Party may seek  …  
*Model said:* <think> Okay, let's tackle this question. The user wants to know if the contract supports, contradicts, or doesn't mention the statement that the Receiving Party must notify the Disclosing Party if they're required by law to disclose Confidential Information.  …

**doc 590 · nda-20** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Gold evidence:* In that case, or if the  
*Predicted evidence:* However, Receiving Party may retain in the office of its legal counsel, one copy of Proprietary Information for record purposes only.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may retain some Confidential Information even after returning or destroying it, based on the provided contract. First, I need to locate the relevant section in the contrac …

**doc 45 · nda-7** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* During this period the receiving party shall not disclose the received “Confidential Information” to any third party, and shall not use the received “Confidential Information” for any purpose other than the aforesaid purposes.  
*Predicted evidence:* 2. Unless otherwise expressly authorized by the disclosing party, the receiving party agrees to retain the "Confidential Information" in confidence for a period of three (3) years from the date of receipt of the "Confidential Information" or until such time as the information no longer qualifies as "Confidential Inform …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party can share some confidential information with third parties like consultants, agents, or advisors. The contract is a Non-Disclosure Agreement (NDA), so I need to check the  …

**doc 406 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* "Proprietary Information" means information including, but not limited to clients, pricing and information which is related to the business of Health Card from which Health Card  
*Predicted evidence:* 1. a. ... with respect to any oral presentation or communication, denominating such information as confidential immediately before, during or after such oral presentation or communication;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Confidential Information may include verbally conveyed information." The contract in question is a Confidentiality and Non-Disclosure Agreement betwe …

**doc 16 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* 2. Recipient acknowledges the sensitive and secret nature of the Confidential Information it will have access to during the term of this Agreement and agrees that it will treat such Confidential Information as strictly confidential and shall exercise the same degree of care in the protection of the Confidential Informa …  
*Predicted evidence:* 10. The representations and obligations of the Recipient relative to the Confidential Information shall survive the termination of this Confidentiality Agreement for a period up to three (3) years from the date of the completion or other termination of the Project.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Some obligations of Agreement may survive termination of Agreement." First, I need to look through the contract to find any clauses that mention obli …

**doc 411 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* (a) Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any business contacts, customers, suppliers, techno …  
*Predicted evidence:* Each of Stockholders severally agrees that such Stockholder shall not... use for his or their own account any business contacts... or other secret or confidential information...  
*Model said:* <think> Okay, let's tackle this query. The user is asking whether the contract supports, contradicts, or doesn't mention the statement: "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement." Fi …

**doc 35 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* For purposes of this Agreement, "Confidential Information" shall include, without limitation, any information or other data relating to the Disclosing Party or any of its Affiliates, which is disclosed to the Receiving Party, whether prior to the conclusion of this Agreement, in terms of this Agreement or otherwise in  …  
*Predicted evidence:* 2.3.2 which includes, without derogating from the aforegoing, all financial information, marketing information, products, product information, drawings, plans and specifications, operational and risk methodologies, production and operating procedures, computer data, programs and source codes, pricing, price lists and p …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that "Confidential Information shall only include technical information." First, I need to look at the contract's definiti …

**doc 29 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* CI does not include information that  
*Predicted evidence:* 3. No Other Rights. ... (A) may independently develop and acquire software, hardware, and other technology for its own products, services, and processes, and that having access to Discloser’s CI does not prohibit Recipient from freely developing or acquiring similar products without obligation to Discloser;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the receiving party (Recipient) may acquire information similar to Confidential Information from a third party. The contract is a Non-Disclosure Agreement (NDA) between Adobe and Acme Corpora …

**doc 405 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* In the event that you or any of your representatives receive a request or are required (by deposition, interrogatory, request for documents, subpoena, civil investigative demand or similar process) to disclose all or any part of the Evaluation Materials, you or your representatives, as the case may be, agree to  
*Predicted evidence:* In the event that you or any of your representatives receive a request or are required (by deposition, interrogatory, request for documents, subpoena, civil investigative demand or similar process) to disclose all or any part of the Evaluation Materials, you or your representatives, as the case may be, agree to (i) imm …  
*Model said:* <think> Okay, let's tackle this query. The user is asking whether the contract supports, contradicts, or doesn't mention the statement: "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to …

**doc 61 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 1. Company agrees that all information, documents, data, materials and the AOC proprietary software systems disclosed to, or accessed by, Company in connection with, or related to, the RFP or the RFP process, in any form whether oral or written, or in any medium, including but not limited to any of the following is “Co …  
*Predicted evidence:* 1. Company agrees that all information, documents, data, materials and the AOC proprietary software systems disclosed to, or accessed by, Company in connection with, or related to, the RFP or the RFP process, in any form whether oral or written, or in any medium, including but not limited to any of the following is "Co …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the statement that "Confidential Information may include verbally conveyed information." I need to check the contract text provided to see if there's any mention of verb …

**doc 586 · nda-18** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not solicit some of Disclosing Party's representatives.  
*Gold evidence:* For a period of one year subsequent to the termination of discussions regarding a Transaction, neither Party shall, without the prior written consent of the other Party, directly or indirectly solicit for hire any person currently employed by the other Party (or any of its subsidiaries) with whom the hiring Party first …  
*Predicted evidence:* 5. No Solicitation. For a period of one year subsequent to the termination of discussions regarding a Transaction, neither Party shall, without the prior written consent of the other Party, directly or indirectly solicit for hire any person currently employed by the other Party (or any of its subsidiaries) with whom th …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Receiving Party shall not solicit some of Disclosing Party's representatives." First, I need to understand the context. The contract is a Mutual Non-D …

**doc 575 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* Vapotherm agrees to disclose the Confidential Information only for the following specific purpose (‘Permitted Purpose”):  
*Predicted evidence:* 2. Purpose. Vapotherm agrees to disclose the Confidential Information only for the following specific purpose (‘Permitted Purpose’): ________________________________________________________________________________________________ __________________________________________________________________________________________ …  
*Model said:* <think> Okay, let's tackle this question. The user wants to know if the contract supports the statement that the Receiving Party shall not use any Confidential Information for any purpose other than those stated in the Agreement. First, I need to look through  …

**doc 72 · nda-15** — gold `Entailment`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Also, BDO reserves the right not to sell or otherwise dispose of the Assets and to terminate, at any time, further participation in the investigation and due diligence process by you or any other party and to modify any data, documentation and other procedures relating to the proposed sale without assigning any reason  …  
*Predicted evidence:* 10. The undersigned acknowledges and agrees that the execution and delivery of this NDA and the delivery of the Information does not give rise to any legal obligation of BDO or Baldwin, whether in contract, in negligence or other tort, by way of fiduciary duty or otherwise. Without limiting the generality of the forego …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Agreement shall not grant Receiving Party any right to Confidential Information." So, I need to check the contract to see if it says that the receivin …


### gemma-4-12B-it (A) — 340 cases, showing 20

**doc 33 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* In consideration of each party supplying its Confidential Information to the other party and subject to the exceptions described in Clause 5, the parties undertake as follows:  
*Predicted evidence:* 2.2 not to use the other party's Confidential Information except for the Purpose;  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["2.2 not to use the other party's Confidential Information except for the Purpose;"]} ```

**doc 205 · nda-20** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.  
*Gold evidence:* Having acknowledged that it, particularly within a networked environment, the return and/or destruction of electronically stored information may be impossible, extremely or otherwise unduly difficult or costly, the immediately preceding sentence is not intended to, and does not obligate the Receiving Party to make extr …  
*Predicted evidence:* 6.4. Notwithstanding Section 6.3, the Receiving Party may retain a reasonable number of copies of the Confidential Information (and any materials embedding the same) (“Retained Copies”) for the sole purpose of (i) satisfying any legal or regulatory requirements regarding record and data retention that the Receiving Par …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["6.4. Notwithstanding Section 6.3, the Receiving Party may retain a reasonable number of copies of the Confidential Information (and any materials embedding the same) (“Retained Copies”) for the sole purpose of (i …

**doc 72 · nda-7** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).  
*Gold evidence:* 2. Not to supply any documents or portions of documents included in the Information or any information included therein or any information hereinafter obtained in the course hereof or with respect hereto to any person or corporation except to the Representatives (defined below) and as otherwise specifically provided fo …  
*Predicted evidence:* 3. The Information is to be used by the undersigned, its affiliates, and their respective directors, officers, employees, accountants, attorneys, solicitors, financing sources, consultants, agents, representatives and advisors and their respective representatives, to the extent the foregoing are actually provided acces …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["3. The Information is to be used by the undersigned, its affiliates, and their respective directors, officers, employees, accountants, attorneys, solicitors, financing sources, consultants, agents, representative …

**doc 598 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* If the Recipient or any of its Representatives is requested to disclose any Confidential Information in connection with any legal or administrative proceeding or investigation, or is required by law, regulation, stock exchange or regulatory authority to disclose any Confidential Information, such person or entity will  
*Predicted evidence:* If the Recipient or any of its Representatives is requested to disclose any Confidential Information in connection with any legal or administrative proceeding or investigation, or is required by law, regulation, stock exchange or regulatory authority to disclose any Confidential Information, such person or entity will  …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["If the Recipient or any of its Representatives is requested to disclose any Confidential Information in connection with any legal or administrative proceeding or investigation, or is required by law, regulation,  …

**doc 33 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* This Agreement does not apply to Confidential Information:  
*Predicted evidence:* 5.3 which the Receiving Party obtains or has available from a source other than the Disclosing Party without breaching any obligation of confidentiality;  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["5.3 which the Receiving Party obtains or has available from a source other than the Disclosing Party without breaching any obligation of confidentiality;"]} ```

**doc 69 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* (b) “ConfidentialInformation” means all information, whether visual, written, electronic or ​ ​ ​  
*Predicted evidence:* (c) “ConfidentialInformation” means all information, whether visual, written, electronic or oral, (including any information furnished prior to the execution of this Agreement) related to the products, services, technology, business, customers, working capital, affairs, assets, condition (financial or otherwise) or pro …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["(c) “ConfidentialInformation” means all information, whether visual, written, electronic or oral, (including any information furnished prior to the execution of this Agreement) related to the products, services,  …

**doc 575 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* “Confidential Information” shall not include:  
*Predicted evidence:* Information that is disclosed by a third party to Receiving Party after the date of disclosure by Vapotherm, where the third party did not require Receiving Party to hold such information in confidence and did not acquire such information directly or indirectly from Vapotherm.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Information that is disclosed by a third party to Receiving Party after the date of disclosure by Vapotherm, where the third party did not require Receiving Party to hold such information in confidence and did no …

**doc 81 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* The receiving party will not be liable for the disclosure of any Confidential Information which is:  
*Predicted evidence:* (d) independently developed by employees and / or independent contactors of the receiving party;  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["(d) independently developed by employees and / or independent contactors of the receiving party;"]} ```

**doc 582 · nda-8** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.  
*Gold evidence:* 4. In the event you or any of your Representatives become legally compelled (by deposition, interrogatory, request for documents, subpoena, civil investigation, demand, order or similar process) to disclose any of the contents of the Confidential Information, or you or the Company or any of the respective Representativ …  
*Predicted evidence:* the Company and you each agrees that each party and its respective Representatives may do so without liability, but each party hereto also agrees (i) to promptly notify the other party prior to any such disclosure to the extent practicable and legally permissible  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["the Company and you each agrees that each party and its respective Representatives may do so without liability, but each party hereto also agrees (i) to promptly notify the other party prior to any such disclosur …

**doc 63 · nda-10** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Gold evidence:* 10. Each Party hereto agrees, and shall cause its Representatives, not to announce or disclose to any third person its participation in discussions with the other party concerning the matters contained in this Agreement without the advance written consent of the other Party.  
*Predicted evidence:* Each Party hereto agrees, and shall cause its Representatives, not to announce or disclose to any third person its participation in discussions with the other party concerning the matters contained in this Agreement without the advance written consent of the other Party.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Each Party hereto agrees, and shall cause its Representatives, not to announce or disclose to any third person its participation in discussions with the other party concerning the matters contained in this Agreem …

**doc 605 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* (b) “Blueknight Non-public Information” shall mean information pertaining to Blueknight, the General Partner and/or their respective Affiliates that has not been Publicly Disclosed and that Blueknight or the General Partner discloses to Receiving Party, including, without limitation, the existence of the Refinancing Me …  
*Predicted evidence:* “Blueknight Non-public Information” shall mean information pertaining to Blueknight, the General Partner and/or their respective Affiliates that has not been Publicly Disclosed and that Blueknight or the General Partner discloses to Receiving Party, including, without limitation, the existence of the Refinancing Meetin …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["“Blueknight Non-public Information” shall mean information pertaining to Blueknight, the General Partner and/or their respective Affiliates that has not been Publicly Disclosed and that Blueknight or the Gener …

**doc 486 · nda-10** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.  
*Gold evidence:* d) This Agreement and each additional agreement concluded or written or verbal disclosure made between the Parties, shall be kept confidential and is not to be reproduced, communicated or distributed in any manner whatsoever except on a "need to know" basis to persons directly involved with the closing of any transacti …  
*Predicted evidence:* The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge and know-how, and will disclose such information only pursuant to the express written permission of the party …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["The Parties will maintain complete confidentiality regarding this Agreement and all transactions occurring thereunder, each other's business, business sources and affiliates and each other's propriety knowledge a …

**doc 605 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Receiving Party shall  
*Predicted evidence:* provided, however, that Receiving Party may disclose such information to such of its representatives who need such information for the purpose of evaluating or assisting with Receiving Party’s investment in Blueknight, it being understood that such representatives shall be informed in advance by Receiving Party of the  …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["provided, however, that Receiving Party may disclose such information to such of its representatives who need such information for the purpose of evaluating or assisting with Receiving Party’s investment in Bluek …

**doc 81 · nda-1** — gold `Contradiction`, predicted `Contradiction`  
*Q:* All Confidential Information shall be expressly identified by the Disclosing Party.  
*Gold evidence:* The confidential, proprietary and trade secret information of the disclosing party (“Confidential Information”) to be disclosed hereunder is  
*Predicted evidence:* However, the disclosing party’s information in tangible form that does not bear any of these legends, and discussions relating to that information, shall nevertheless be protected hereunder as Confidential Information, if the receiving party knew, or should have reasonably known under the circumstances, that the inform …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["However, the disclosing party’s information in tangible form that does not bear any of these legends, and discussions relating to that information, shall nevertheless be protected hereunder as Confidential Inf …

**doc 39 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* By signing this agreement the Bidder is stating that it will keep all Confidential Information (as defined in paragraph 4) in strict confidence and only use the Confidential Information on the Project identified herein.  
*Predicted evidence:* The purpose of this agreement is to allow the Bidder access to Confidential Information as defined herein detailing the design of the Project for the sole purpose of compiling and submitting a bid in response to the County’s Invitation for Bid to construct the Project.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["The purpose of this agreement is to allow the Bidder access to Confidential Information as defined herein detailing the design of the Project for the sole purpose of compiling and submitting a bid in response to  …

**doc 64 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* The Retained Information shall be held by the Receiving Party, and kept confidential in strict accordance with the terms of this Confidentiality Agreement, or shall be destroyed, unless the Receiving Party is prohibited from destroying it.  
*Predicted evidence:* The Information shall be kept confidential in perpetuity, and in strict accordance with the provisions of this Confidentiality Agreement.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["The Information shall be kept confidential in perpetuity, and in strict accordance with the provisions of this Confidentiality Agreement."]} ```

**doc 73 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* (a) “Confidential Information” shall include all information pertaining to the Assignors and their respective affiliates, employees, salespersons, independent contractors, agents, officers, directors, customers, suppliers, vendors, manufacturers, distributors, dealers, predecessors, successors and assigns disclosed by  …  
*Predicted evidence:* (including, but not limited to, information learned by the Receiving Party from the Disclosing Party’s employees, agents, officers, directors, professional advisors or other representatives or through inspection of the Assigned Assets or the books and records of the Assignors)  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["(including, but not limited to, information learned by the Receiving Party from the Disclosing Party’s employees, agents, officers, directors, professional advisors or other representatives or through inspection  …

**doc 405 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* You and your representatives (as defined herein) will treat confidentially any information (whether written or oral) that either we or our financial advisor, Credit Suisse First Boston Corporation ("CSFB"), or our other representatives furnish to you in connection with a Possible Transaction involving the Company, toge …  
*Predicted evidence:* The term "Evaluation Materials" includes information furnished to you orally or in writing (whatever the form or storage medium) or gathered by inspection, and regardless of whether such information is specifically identified as "confidential".  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["The term \"Evaluation Materials\" includes information furnished to you orally or in writing (whatever the form or storage medium) or gathered by inspection, and regardless of whether such information is specific …

**doc 12 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* 1.2 BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not having reasonable need for access to such information, unless CLIENT expressly consents to such disclo …  
*Predicted evidence:* BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not having reasonable need for access to such information, unless CLIENT expressly consents to such disclosure …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS no …

**doc 507 · nda-4** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* In consideration of Big Sky's disclosure to Holdings of the Information, Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectiv …  
*Predicted evidence:* Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectively, "Representatives"), in any manner whatsoever, in whole or in part, a …  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, a …


### granite-4.2-8b (A) — 220 cases, showing 20

**doc 69 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* (f) “Representatives” means all Persons acting on behalf of a specified Party or its Affiliates, including without limitation, its directors, officers, lenders, employees, agents, representatives, financial advisors, attorneys, accountants, consultants, reserve and valuation and other experts.  
*Predicted evidence:* Each Receiving Party shall keep all Confidential Information it receives or obtains from a Disclosing Party confidential and shall not, without the prior written consent of the Disclosing Party, disclose such Confidential Information, in whole or in part, and shall not use such Confidential Information, directly or ind …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Each Receiving Party shall keep all Confidential Information it receives or obtains from a Disclosing Party confidential and shall not, without the prior written consent of the Disclosing Party, disclose such Confident …

**doc 72 · nda-15** — gold `Entailment`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* Also, BDO reserves the right not to sell or otherwise dispose of the Assets and to terminate, at any time, further participation in the investigation and due diligence process by you or any other party and to modify any data, documentation and other procedures relating to the proposed sale without assigning any reason  …  
*Predicted evidence:* The undersigned acknowledges and agrees that the execution and delivery of this NDA and the delivery of the Information does not give rise to any legal obligation of BDO or Baldwin, whether in contract, in negligence or other tort, by way of fiduciary duty or otherwise.  
*Model said:* { "verdict": "Entailment", "evidence": [ "The undersigned acknowledges and agrees that the execution and delivery of this NDA and the delivery of the Information does not give rise to any legal obligation of BDO or Baldwin, whether in contract, in negligence o …

**doc 45 · nda-15** — gold `Entailment`, predicted `Entailment`  
*Q:* Agreement shall not grant Receiving Party any right to Confidential Information.  
*Gold evidence:* 6. No license, express or implied, in the information disclosed is granted to either party other than to use the information in the manner and to the extent authorized by the Agreement.  
*Predicted evidence:* No license, express or implied, in the information disclosed is granted to either party other than to use the information in the manner and to the extent authorized by the Agreement.  
*Model said:* { "verdict": "Entailment", "evidence": [ "No license, express or implied, in the information disclosed is granted to either party other than to use the information in the manner and to the extent authorized by the Agreement.", "Nothing herein shall be construe …

**doc 81 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* The receiving party will not disclose any of the disclosing party’s Confidential Information to any employees or to any third parties except to the receiving party’s employees, parent company, affiliates and subsidiaries who have a need to know and who agree to abide by nondisclosure terms at least as comprehensive as  …  
*Predicted evidence:* The receiving party will not disclose any of the disclosing party’s Confidential Information to any employees or to any third parties except to the receiving party’s employees, parent company, affiliates and subsidiaries who have a need to know and who agree to abide by nondisclosure terms at least as comprehensive as  …  
*Model said:* { "verdict": "Entailment", "evidence": [ "The receiving party will not disclose any of the disclosing party’s Confidential Information to any employees or to any third parties except to the receiving party’s employees, parent company, affiliates and subsidiari …

**doc 45 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* 2. Unless otherwise expressly authorized by the disclosing party, the receiving party agrees to retain the “Confidential Information” in confidence for a period of three (3) years from the date of receipt of the “Confidential Information” or until such time as the information no longer qualifies as “Confidential Inform …  
*Predicted evidence:* 4. Either party may terminate this agreement at any time by giving written notice to the other party. Notwithstanding termination or expiration, all obligations of confidentially shall survive and continue to bind the receiving parties for the period set forth in Section 2.  
*Model said:* { "verdict": "Entailment", "evidence": [ "4. Either party may terminate this agreement at any time by giving written notice to the other party. Notwithstanding termination or expiration, all obligations of confidentially shall survive and continue to bind the  …

**doc 15 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 3. The receiving party will receive all confidential information (whether recorded in writing or by other means or given orally without record) which is disclosed in connection with this Agreement subject to the following conditions:  
*Predicted evidence:* In general the receiving party must afford disclosed confidential information the same degree of protection as it would afford its own.  
*Model said:* { "verdict": "Entailment", "evidence": [ "In general the receiving party must afford disclosed confidential information the same degree of protection as it would afford its own.", "The receiving party will receive all confidential information (whether recorded …

**doc 586 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* Notwithstanding the foregoing, the term "Evaluation Material" shall not include, and the Parties' obligations herein (other than their obligations under paragraph 6 of this Agreement) shall not extend to, information which  
*Predicted evidence:* 9. Miscellaneous. (a) No License. Neither Party grants a license, by implication or otherwise, under any of its trade secrets or other intellectual property rights to the Receiving Party. The terms of this Agreement shall not be construed to limit either Party's right to independently develop or acquire products withou …  
*Model said:* { "verdict": "Entailment", "evidence": [ "9. Miscellaneous.\n(a) No License. Neither Party grants a license, by implication or otherwise, under any of its trade secrets or other intellectual property rights to the Receiving Party. The terms of this Agreement s …

**doc 29 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* CI does not include information that  
*Predicted evidence:* Each party agrees that Recipient (A) may independently develop and acquire software, hardware, and other technology for its own products, services, and processes, and that having access to Discloser’s CI does not prohibit Recipient from freely developing or acquiring similar products without obligation to Discloser;  
*Model said:* { "verdict": "Entailment", "evidence": [ "Each party agrees that Recipient (A) may independently develop and acquire software, hardware, and other technology for its own products, services, and processes, and that having access to Discloser’s CI does not prohi …

**doc 405 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* You and your representatives (as defined herein) will treat confidentially any information (whether written or oral) that either we or our financial advisor, Credit Suisse First Boston Corporation ("CSFB"), or our other representatives furnish to you in connection with a Possible Transaction involving the Company, toge …  
*Predicted evidence:* The term "Evaluation Materials" includes information furnished to you orally or in writing (whatever the form or storage medium) or gathered by inspection, and regardless of whether such information is specifically identified as "confidential".  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The term \"Evaluation Materials\" includes information furnished to you orally or in writing (whatever the form or storage medium) or gathered by inspection, and regardless of whether such information is specificall …

**doc 605 · nda-5** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may share some Confidential Information with some of Receiving Party's employees.  
*Gold evidence:* Receiving Party shall  
*Predicted evidence:* Receiving Party may disclose such information to such of its representatives who need such information for the purpose of evaluating or assisting with Receiving Party’s investment in Blueknight, it being understood that such representatives shall be informed in advance by Receiving Party of the confidential nature of s …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Receiving Party may disclose such information to such of its representatives who need such information for the purpose of evaluating or assisting with Receiving Party’s investment in Blueknight, it being understood tha …

**doc 75 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* Any and all information, including but not limited to, location, financial information, title reports, surveys, and other agreements, documents, materials and oral and/or written information with respect to the Property, whether delivered or made available concurrently with this Agreement or to be delivered in the futu …  
*Predicted evidence:* Any and all information, including but not limited to, location, financial information, title reports, surveys, and other agreements, documents, materials and oral and/or written information with respect to the Property, whether delivered or made available concurrently with this Agreement or to be delivered in the futu …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Any and all information, including but not limited to, location, financial information, title reports, surveys, and other agreements, documents, materials and oral and/or written information with respect to the Prop …

**doc 3 · nda-11** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.  
*Gold evidence:* The Receiving Parties undertake and agree:  
*Predicted evidence:* Not to alter, modify, disassemble, reverse engineer or decompile any Confidential Information;  
*Model said:* { "verdict": "Entailment", "evidence": [ "Not to alter, modify, disassemble, reverse engineer or decompile any Confidential Information;" ] }

**doc 82 · nda-19** — gold `Entailment`, predicted `Entailment`  
*Q:* Some obligations of Agreement may survive termination of Agreement.  
*Gold evidence:* You acknowledge and confirm as follows:  
*Predicted evidence:* 2.4 the provisions of this Agreement shall continue in effect notwithstanding any decision not to proceed with any possible transaction relating to the Business or any return or destruction of the Confidential Information.  
*Model said:* { "verdict": "Entailment", "evidence": [ "2.4 the provisions of this Agreement shall continue in effect notwithstanding any decision not to proceed with any possible transaction relating to the Business or any return or destruction of the Confidential Informat …

**doc 488 · nda-13** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may acquire information similar to Confidential Information from a third party.  
*Gold evidence:* 4. There shall be no liability for disclosure or use of Confidential Information which is  
*Predicted evidence:* Except as set forth in this Section 2, all Confidential Information shall be in tangible form and shall be marked as Confidential or proprietary information of the disclosing party. If the Confidential Information is disclosed orally or visually, it shall be identified as such at the time of disclosure and confirmed in …  
*Model said:* { "verdict": "Entailment", "evidence": [ "Except as set forth in this Section 2, all Confidential Information shall be in tangible form and shall be marked as Confidential or proprietary information of the disclosing party. If the Confidential Information is d …

**doc 598 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 1. Confidential Information: “Confidential Information” means all information which is non-public, confidential or proprietary in nature, whether transferred in writing, orally, visually, electronically or by other means, disclosed by the Corporation or its representatives to the Recipient on or after the date hereof a …  
*Predicted evidence:* Confidential Information means all information which is non-public, confidential or proprietary in nature, whether transferred in writing, orally, visually, electronically or by other means, disclosed by the Corporation or its representatives to the Recipient on or after the date hereof and for the Purpose.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Confidential Information means all information which is non-public, confidential or proprietary in nature, whether transferred in writing, orally, visually, electronically or by other means, disclosed by the Corporatio …

**doc 435 · nda-4** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.  
*Gold evidence:* 5. "Residual Information" shall mean any Confidential Information of the disclosing party which may be retained in intangible form in the minds of those individuals of the receiving party who have had proper access to such Confidential Information.  
*Predicted evidence:* 3. Each of the parties agrees that it will not make use of, disseminate, or in any way disclose any Confidential Information of the other party to any person, firm or business, except to the extent necessary for negotiations, discussions and consultations with personnel or authorized representatives of the other party  …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "3. Each of the parties agrees that it will not make use of, disseminate, or in any way disclose any Confidential Information of the other party to any person, firm or business, except to the extent necessary for neg …

**doc 7 · nda-16** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.  
*Gold evidence:* Each party shall:  
*Predicted evidence:* 3. RETURN OF CONFIDENTIAL INFORMATION Each party shall: (a) within one (1) month of completion of the purpose or receipt of a written request from the other party, return to the other party all documents and materials (and all copies thereof) containing the other party's Confidential Information and certify in writing  …  
*Model said:* { "verdict": "Entailment", "evidence": [ "3. RETURN OF CONFIDENTIAL INFORMATION\nEach party shall:\n(a) within one (1) month of completion of the purpose or receipt of a written request from the other party, return to the other party all documents and material …

**doc 81 · nda-12** — gold `Entailment`, predicted `Entailment`  
*Q:* Receiving Party may independently develop information similar to Confidential Information.  
*Gold evidence:* The receiving party will not be liable for the disclosure of any Confidential Information which is:  
*Predicted evidence:* 4. Exceptions to the Obligation of Confidentiality. The receiving party will not be liable for the disclosure of any Confidential Information which is: ... (d) independently developed by employees and / or independent contactors of the receiving party;  
*Model said:* { "verdict": "Entailment", "evidence": [ "4. Exceptions to the Obligation of Confidentiality. The receiving party will not be liable for the disclosure of any Confidential Information which is: ... (d) independently developed by employees and / or independent  …

**doc 563 · nda-2** — gold `Contradiction`, predicted `Contradiction`  
*Q:* Confidential Information shall only include technical information.  
*Gold evidence:* 1. “Confidential Information” means, for each party, whether in oral, written or in electronic form:  
*Predicted evidence:* 1. “Confidential Information” means, for each party, whether in oral, written or in electronic form: (i) any information, data, intellectual property (including trade secrets), facts and financial information relating to the existing or proposed business, products, customers or services of such party and its affiliates …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1. “Confidential Information” means, for each party, whether in oral, written or in electronic form: (i) any information, data, intellectual property (including trade secrets), facts and financial information relati …

**doc 16 · nda-3** — gold `Entailment`, predicted `Entailment`  
*Q:* Confidential Information may include verbally conveyed information.  
*Gold evidence:* 1. Recipient may be provided with access to selected information (oral, written, or electronic) of the other (including of any of the Corporations) which helps meet a regulatory purpose or business need, including, but not limited to, that information which relates or refers to: business planning; internal controls; co …  
*Predicted evidence:* This information received from a party (including of any of the Corporations), either orally or in writing, during the course of the Project shall be deemed to be confidential information (Confidential Information) for purposes of this Confidentiality Agreement.  
*Model said:* { "verdict": "Entailment", "evidence": [ "This information received from a party (including of any of the Corporations), either orally or in writing, during the course of the Project shall be deemed to be confidential information (Confidential Information) for …


## 7. Carve-out failures — worked examples

Wrong-verdict cases where the gold clause sits in exception language.


### Qwen3-14B (A) — 88 cases, showing 8

**doc 152 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 3. When receiving the Information each party agrees to maintain all such Information in strict confidence and to use at least the same measures to protect the same as that party uses to protect its own confidential information, including but not limited to:  
*Model said:* {"verdict": "Contradiction", "evidence": ["3. When receiving the Information each party agrees to maintain all such Information in strict confidence and to use at least the same measures to protect the same as that party uses to protect its …

**doc 610 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* As a condition to such information being furnished to each party, its affiliates, its subsidiaries and other individuals and entities controlled, directly or indirectly, by such party (“Affiliates”), and the respective directors, employees, consultants, accountants, attorneys and advisors of such party and its Affiliat …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Neither party will disclose any Confidential Information of the other party to third parties except those of its Associates who are required to have the information in order to evaluate and discuss …

**doc 13 · nda-12** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* Confidential Information excludes any information that:  
*Model said:* {"verdict": "NotMentioned", "evidence": []}

**doc 15 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 2. The parties will mark or otherwise designate confidential information to show expressly or by necessary implication that it is imparted in confidence.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The parties will mark or otherwise designate confidential information to show expressly or by necessary implication that it is imparted in confidence." ] }

**doc 398 · nda-4** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* (b) Seller will keep confidential and will not for a period of three (3) years after the execution of this Agreement, directly or indirectly, divulge to anyone, use or otherwise appropriate any of the Confidential Information for any reason or purpose whatsoever except to authorized representatives of Purchasers or the …  
*Model said:* {"verdict": "NotMentioned", "evidence": []}

**doc 32 · nda-13** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* The obligation of confidentiality set forth in this Agreement shall not apply to any information that:  
*Model said:* {"verdict": "NotMentioned", "evidence": []}

**doc 440 · nda-17** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* 5. The Receiving Party shall not reverse engineer, disassemble, decompile, or otherwise analyze, restructure and/or copy or in any way recreate in whole or in part any Confidential Information of the Disclosing Party.  
*Model said:* {"verdict": "Entailment", "evidence": ["6. If any party provides any of the other parties with written notification of its request for return of its Confidential Information, then the other party(ies) shall promptly return (unless destroyed …

**doc 10 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* Except as set forth in Section 2.06, the Receiving Party shall not disclose any of the Confidential Information or other facts directly related to the Confidential Information to any Person other than its Representatives who have a need to know such Confidential Information and who have been informed of the confidentia …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2.05 No Disclosure: Except as set forth in Section 2.06, the Receiving Party shall not disclose any of the Confidential Information or other facts directly related to the Confidential Information …


### Qwen3-14B (B) — 63 cases, showing 8

**doc 595 · nda-13** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* The term “Evaluation Material” does not include any information which  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (which would be FLIR Systems, Inc. in this contract) may acquire information similar to Confidential Information from a third party. The contract in qu …

**doc 7 · nda-13** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* Confidential Information does not include information:  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party (which would be the Company in this NDA) may acquire information similar to Confidential Information from a third party. The contract needs to be check …

**doc 435 · nda-4** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* 5. "Residual Information" shall mean any Confidential Information of the disclosing party which may be retained in intangible form in the minds of those individuals of the receiving party who have had proper access to such Confidential Information.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract states that the receiving party shouldn't use Confidential Information for any purpose other than those stated in the agreement. First, I need to look through …

**doc 39 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* The Bidder shall not publish, or make and/or retain copies or disclose to any third party, any Confidential Information furnished to the Bidder except to such extent as may be necessary to carry out the Bidder’s duties under this agreement and except to the extent that disclosure is required by judicial process in a le …  
*Model said:* <think> Okay, let's see. The user is asking whether the contract allows the Receiving Party (which is the Bidder here) to share some Confidential Information with third parties like consultants, agents, and professional advisors. Looking at …

**doc 7 · nda-17** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* In consideration of the mutual exchange and disclosure of Confidential Information, each party undertakes in relation to the other party's Confidential Information:  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party (which would be the Company in this contract) can create a copy of some Confidential Information under certain circumstances. Looking at the contract, specifically Clau …

**doc 205 · nda-10** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 1.1. “Confidential Information” shall mean information that is revealed by or obtained through a party (whether in writing, orally or by any other means) (a “Disclosing Party”) to the other (a “Receiving Party”) including, without limitation,  
*Model said:* <think> Okay, let's see. The user is asking whether the Receiving Party is prohibited from disclosing the fact that the Agreement was agreed or negotiated. I need to check the contract provided to see if there's any mention of this. Looking …

**doc 610 · nda-4** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* Each of the parties hereto agree not to use the Confidential Information disclosed to it by the other party or its Associates except in connection with preparing a proposal for and any discussions of a Transaction or as otherwise permitted hereunder.  
*Model said:* <think> Okay, let's tackle this question. The user wants to know if the contract supports the statement that the Receiving Party shall not use any Confidential Information for any purpose other than those stated in the Agreement. First, I n …

**doc 16 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 3. Recipient shall not disclose Confidential Information to any person or entity, except its employees or partners involved in the Project, or to use the Confidential Information for any purpose other than fulfilling its obligations under the Project.  
*Model said:* <think> Okay, let's see. The user is asking whether the contract allows the Receiving Party (Recipient) to share Confidential Information with third parties like consultants, agents, or professional advisors. Looking at the contract, specif …


### Qwen3-4B (A) — 114 cases, showing 8

**doc 81 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* Title or the right to possess Confidential Information as between the parties will remain in the disclosing party.  
*Model said:* {"verdict": "Contradiction", "evidence": ["5. Title. Title or the right to possess Confidential Information as between the parties will remain in the disclosing party."]}

**doc 610 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* As a condition to such information being furnished to each party, its affiliates, its subsidiaries and other individuals and entities controlled, directly or indirectly, by such party (“Affiliates”), and the respective directors, employees, consultants, accountants, attorneys and advisors of such party and its Affiliat …  
*Model said:* {"verdict": "Contradiction", "evidence": ["A party disclosing Confidential Information hereunder is referred to herein as \"the disclosing party\" and a party receiving the Confidential Information of a disclosing party hereunder is referre …

**doc 507 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* In consideration of Big Sky's disclosure to Holdings of the Information, Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, directors, employees, agents or representatives (collectiv …  
*Model said:* {"verdict": "Contradiction", "evidence": ["Holdings agrees that it will keep the Information confidential and that the Information will not, without the prior written consent of Big Sky, be disclosed by Holdings or by its officers, director …

**doc 45 · nda-10** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 8. The existence of this Agreement and/or the nature of the business conducted between the parties shall not be disclosed at any time by any party(ies) without the prior written permission of the other party.  
*Model said:* {"verdict": "NotMentioned", "evidence": []}

**doc 563 · nda-10** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 1. “Confidential Information” means, for each party, whether in oral, written or in electronic form:  
*Model said:* {"verdict": "Contradiction", "evidence": ["5. Except as provided in Section 3, the Recipient will not, without the consent of the Disclosing Party, (i) disclose to any person the fact that it has obtained the Confidential Information, or (i …

**doc 45 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 6. No license, express or implied, in the information disclosed is granted to either party other than to use the information in the manner and to the extent authorized by the Agreement.  
*Model said:* {"verdict": "Contradiction", "evidence": ["6. No license, express or implied, in the information disclosed is granted to either party other than to use the information in the manner and to the extent authorized by the Agreement."]}

**doc 575 · nda-17** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* The Receiving Party further agrees:  
*Model said:* {"verdict": "Entailment", "evidence": ["d. Not to copy Confidential Information in whole or in part, without the express written permission of Vapotherm;"]}

**doc 12 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 1.2 BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employee of BROOKS not having reasonable need for access to such information, unless CLIENT expressly consents to such disclo …  
*Model said:* {"verdict": "Contradiction", "evidence": ["BROOKS, its directors, employees, agents and representatives shall not disclose, disseminate, publish, communicate or divulge any Confidential Information to anyone outside BROOKS, or to any employ …


### Qwen3-4B (B) — 90 cases, showing 8

**doc 64 · nda-19** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* The Retained Information shall be held by the Receiving Party, and kept confidential in strict accordance with the terms of this Confidentiality Agreement, or shall be destroyed, unless the Receiving Party is prohibited from destroying it.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that some obligations of the Agreement may survive termination. The answer needs to be either Entailment, Contradiction, or NotMentioned …

**doc 17 · nda-17** — gold `Entailment`, predicted `None`  
*Gold evidence:* f. The contractor and its employees shall not make any copies of any VA information, including software or applications that are not copyrighted, except as necessary to restore VA computer systems or data storage devices to an operating state.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which I assume is the contractor) to create a copy of some Confidential Information under certain circumstances. The answer needs  …

**doc 72 · nda-5** — gold `Entailment`, predicted `None`  
*Gold evidence:* 2. Not to supply any documents or portions of documents included in the Information or any information included therein or any information hereinafter obtained in the course hereof or with respect hereto to any person or corporation except to the Representatives (defined below) and as otherwise specifically provided fo …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to share some Confidential Information with some of their employees. The contract in question is a Confidentiality and Non-Disclosu …

**doc 605 · nda-5** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* Receiving Party shall  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party can share some confidential information with their employees based on the provided contract. First, I need to look through the contract text, specifica …

**doc 69 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* The exposure of the Receiving Party to the Confidential Information or its disclosure shall not be construed in any way to grant to the Receiving Party any right or license with respect to the Information other than the right to use such Confidential Information strictly in accordance with this Agreement.  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Agreement shall not grant the Receiving Party any right to Confidential Information. The answer needs to be either Entailment,  …

**doc 16 · nda-13** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 7. The obligation of non-disclosure shall not extend to:  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that the Receiving Party (Recipient) may acquire information similar to Confidential Information from a third party. The answer needs to …

**doc 547 · nda-12** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* If such protective order or other remedy is not obtained and the Disclosing Party has not waived compliance with the provisions of this Agreement, the Receiving Party will furnish only that portion of the Confidential Information which the Receiving Party is advised by opinion of counsel it is legally required to furni …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party to independently develop information similar to Confidential Information. The answer options are Entailment, Contradiction, or NotM …

**doc 39 · nda-2** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* Within ten (10) days after award and/or completion of the IFB, or upon demand by the County, all Confidential Information, including all photographs, sketches, models, and drawings related to the design, layout and operation of said project shall be destroyed and files deleted from all of the Bidder’s web sites and ele …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports the position that "Confidential Information shall only include technical information." The answer needs to be either Entailment, Contradiction, or No …


### Qwen3-8B (A) — 72 cases, showing 8

**doc 482 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 1. "Confidential Information" means all information provided by either Party in writing or visual form which Party desires to protect from disclosure and which is marked with an appropriate stamp or legend designating such material as "Confidential"; and all information provided orally by a Party which that Party desir …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1. \"Confidential Information\" means all information provided by either Party in writing or visual form which Party desires to protect from disclosure and which is marked with an appropriate sta …

**doc 73 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* The term “Confidential Information” shall not include such portions of the Confidential Information that:  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The Receiving Party: (i) shall treat the Confidential Information and the Confidential Materials as confidential in accordance with the terms and conditions of this Agreement; (ii) shall not use  …

**doc 610 · nda-4** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* Each of the parties hereto agree not to use the Confidential Information disclosed to it by the other party or its Associates except in connection with preparing a proposal for and any discussions of a Transaction or as otherwise permitted hereunder.  
*Model said:* { "verdict": "Entailment", "evidence": [ "Each of the parties hereto agree not to use the Confidential Information disclosed to it by the other party or its Associates except in connection with preparing a proposal for and any discussions o …

**doc 70 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* I understand that I may access and/or use FAU Confidential Information, Restricted Data and Sensitive Data only as necessary to perform my job-related duties and that I may disclose (i.e., share) FAU Confidential Information, Restricted Data, and/or Sensitive Data only to authorized individuals with a need to know that …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "I will not disclose any PHI to any individual or third party, except as specifically authorized by FAU policies and procedures, and upon receiving a written authorization from the patient (unless …

**doc 595 · nda-10** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* As used in this letter agreement (this “Agreement”), the term “Evaluation Material” means all information regarding the Company, its businesses, technology, products, prospects and plans, a Transaction or the fact that the Company may pursue a Transaction with you or other parties, the existence and terms of this Lette …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Non-Disclosure: Unless otherwise required by applicable law in the reasonable written opinion of your legal counsel, but only after compliance with the paragraph titled \"Compelled Disclosure\" b …

**doc 406 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 3. Disclosures of the Proprietary  
*Model said:* { "verdict": "Contradiction", "evidence": [ "3. Disclosures of the Proprietary Information may be made only to employees, agents, associates or independent contractors of Ciufo (a) who are directly involved in the Project and have a specifi …

**doc 10 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* Except as set forth in Section 2.06, the Receiving Party shall not disclose any of the Confidential Information or other facts directly related to the Confidential Information to any Person other than its Representatives who have a need to know such Confidential Information and who have been informed of the confidentia …  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Except as set forth in Section 2.06, the Receiving Party shall not disclose any of the Confidential Information or other facts directly related to the Confidential Information to any Person other …

**doc 10 · nda-4** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* The Receiving Party shall not, in any manner, derive any benefit, directly or indirectly, from the Confidential Information or the use of such Confidential Information, for any purpose.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "The Receiving Party shall not, in any manner, derive any benefit, directly or indirectly, from the Confidential Information or the use of such Confidential Information, for any purpose." ] }


### Qwen3-8B (B) — 87 cases, showing 8

**doc 37 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* The Receiving Party acknowledges and agrees that all property, including intellectual property rights, in Confidential Information disclosed to it by the Disclosing Party shall remain with and be vested in the Disclosing Party or relevant member of the Disclosing Party Group.  
*Model said:* <think> Okay, let's tackle this question. The user wants to know if the contract supports the position that "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to look through the contract to fin …

**doc 435 · nda-8** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 4. There shall be no liability for disclosure or use of Confidential Information which is  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that the Receiving Party must notify the Disclosing Party if they're required by law to disclose Confi …

**doc 595 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* You agree that the Evaluation Material is and shall remain the property of the Company or its subsidiaries and affiliates, as applicable, and that neither the Company nor any of its Representatives, subsidiaries or affiliates has granted you any license, copyright, or similar right with respect to any of the Evaluation …  
*Model said:* <think> Okay, let's tackle this query. The user wants to know if the contract supports, contradicts, or doesn't mention the statement: "Agreement shall not grant Receiving Party any right to Confidential Information." First, I need to under …

**doc 64 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* B. “Delivering Party” has furnished, is furnishing, or will be furnishing, either directly or indirectly through a representative, to the other party (“Receiving Party”) or to the Receiving Party’s representatives, including without limitation, any agents, brokers, lender, appraisers, escrow, professional advisors, con …  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party can share some Confidential Information with third parties like consultants, agents, and professional advisors. The contract in question is a Commercia …

**doc 405 · nda-13** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* The term "Evaluation Materials" does not include information which  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which would be Richard A. Bearse in this case) to acquire information similar to Confidential Information from a third party. Firs …

**doc 456 · nda-2** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 4. Consistent with other provisions herein, each party assures that it will not knowingly, without obtaining prior authorization from the U.S. Dept. of Commerce Office of Export Administration, transmit directly or indirectly the technical data received pursuant hereto or the immediate product (including processes and  …  
*Model said:* <think> Okay, let's tackle this. The user is asking whether the contract supports, contradicts, or doesn't mention the statement that "Confidential Information shall only include technical information." First, I need to look through the pro …

**doc 15 · nda-17** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* e) it will not make any copy of or abstract of the information without specific written permission from the disclosing party;  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the Receiving Party may create a copy of some Confidential Information in some circumstances based on the provided contract. First, I need to look through the contract条款 t …

**doc 595 · nda-13** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* The term “Evaluation Material” does not include any information which  
*Model said:* <think> Okay, let's tackle this question. The user is asking whether the contract allows the Receiving Party (which I assume is the party receiving the Evaluation Material) to acquire information similar to Confidential Information from a t …


### gemma-4-12B-it (A) — 54 cases, showing 8

**doc 411 · nda-4** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* (a) Each of Stockholders severally agrees that such Stockholder shall not, at any time following the effectiveness of the Merger, print, publish, divulge or communicate to any person, firm, corporation or other business organization or use for his or their own account any business contacts, customers, suppliers, techno …  
*Model said:* ```json {"verdict": "NotMentioned", "evidence": []} ```

**doc 81 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* Title or the right to possess Confidential Information as between the parties will remain in the disclosing party.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Title or the right to possess Confidential Information as between the parties will remain in the disclosing party."]} ```

**doc 435 · nda-15** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 8. All materials (including, without limitation, documents, drawings, models, apparatus, sketches, designs and lists) furnished to one party by the other, and which are designated in writing to be the property of such party, shall remain the property of such party and shall be returned to it promptly at its request, to …  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Neither party acquires any licenses under any intellectual property rights of the other party under this Agreement."]} ```

**doc 72 · nda-17** — gold `Contradiction`, predicted `Entailment`  
*Gold evidence:* 9. The Information being made available is confidential and may not be photocopied, reproduced or distributed to any other person at any time, except for the purposes of obtaining professional advice, without the prior written consent of BDO.  
*Model said:* ```json {"verdict": "Entailment", "evidence": ["1. To maintain the Information in the strictest of confidence and to control the dissemination of the Information, including any documents or copies (paper, electronic or otherwise) thereof co …

**doc 456 · nda-4** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 4. Consistent with other provisions herein, each party assures that it will not knowingly, without obtaining prior authorization from the U.S. Dept. of Commerce Office of Export Administration, transmit directly or indirectly the technical data received pursuant hereto or the immediate product (including processes and  …  
*Model said:* ```json {"verdict": "NotMentioned", "evidence": []} ```

**doc 456 · nda-15** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 2. No rights or obligations other than expressly recited herein are to be implied here from.  
*Model said:* ```json {"verdict": "NotMentioned", "evidence": []} ```

**doc 71 · nda-5** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* Prospect’s and Prospect’s Broker’s dissemination of any information and materials provided by Listing Broker will be limited to attorneys, accountants, banking representatives, and business advisors directly involved with the above-referenced Property.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["Prospect’s and Prospect’s Broker’s dissemination of any information and materials provided by Listing Broker will be limited to attorneys, accountants, banking representatives, and business …

**doc 15 · nda-1** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 2. The parties will mark or otherwise designate confidential information to show expressly or by necessary implication that it is imparted in confidence.  
*Model said:* ```json {"verdict": "Contradiction", "evidence": ["2. The parties will mark or otherwise designate confidential information to show expressly or by necessary implication that it is imparted in confidence."]} ```


### granite-4.2-8b (A) — 96 cases, showing 8

**doc 28 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 4. The restrictions of this Agreement on use and disclosure of Confidential Information shall not apply to information, which is:  
*Model said:* { "verdict": "Contradiction", "evidence": [ "4. The restrictions of this Agreement on use and disclosure of Confidential Information shall not apply to information, which is:\n(a) in the public domain as of the date of this Agreement or whi …

**doc 69 · nda-7** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* (f) “Representatives” means all Persons acting on behalf of a specified Party or its Affiliates, including without limitation, its directors, officers, lenders, employees, agents, representatives, financial advisors, attorneys, accountants, consultants, reserve and valuation and other experts.  
*Model said:* { "verdict": "Contradiction", "evidence": [ "Each Receiving Party shall keep all Confidential Information it receives or obtains from a Disclosing Party confidential and shall not, without the prior written consent of the Disclosing Party,  …

**doc 205 · nda-10** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* 1.1. “Confidential Information” shall mean information that is revealed by or obtained through a party (whether in writing, orally or by any other means) (a “Disclosing Party”) to the other (a “Receiving Party”) including, without limitation,  
*Model said:* { "verdict": "Contradiction", "evidence": [ "2.2. The Receiving Party shall not disclose, duplicate, transmit or otherwise disseminate in any manner the Disclosing Party’s Confidential Information, except to the Receiving Party’s officers,  …

**doc 37 · nda-13** — gold `Entailment`, predicted `Contradiction`  
*Gold evidence:* “Confidential Information” means:  
*Model said:* { "verdict": "Contradiction", "evidence": [ "1.1(d) but not including any Information that: 1.1.1.2 the Receiving Party obtained on a non-confidential basis from a third party who is not, to the Receiving Party’s knowledge or belief, bound  …

**doc 398 · nda-19** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* (c) With respect to any trade secrets included in the Confidential Information, Seller also agrees not to use or disclose any of such trade secrets at any time until such trade secrets become generally available to the public by independent discovery or development and publication through no fault of Seller.  
*Model said:* { "verdict": "NotMentioned", "evidence": [] }

**doc 75 · nda-19** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* Investor shall keep the Confidential Information permanently and strictly confidential, and not to disclose or divulge any Confidential Information to, or reproduce any Confidential Information for the benefit of any person or entity other than the Investor’s employees, representatives, consultants, and agents who are  …  
*Model said:* { "verdict": "NotMentioned", "evidence": [] }

**doc 406 · nda-13** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* b. Proprietary Information does not include any information which  
*Model said:* { "verdict": "NotMentioned", "evidence": [] }

**doc 13 · nda-5** — gold `Entailment`, predicted `NotMentioned`  
*Gold evidence:* 9. “Reviewing Representative” means an employee of a Party who has signed or electronically signed a Non-Disclosure Certificate and who is a principal, partner, officer, director, employee, agent and other representative of a Party.  
*Model said:* { "verdict": "NotMentioned", "evidence": [] }

