Text mining and NLP using Python libraries
================================

This is the repo for all tests and developments using Python for TM&NLP tasks on Privacy Policy. <br>

List of Content
================================
**01_test_classifiers.py** - tests various classifiers available in scikit-learn.<br>
***02_two_layered_classifier*** - Two layered classifier, with a hard-coded example test for now.<br>
**03_plot_randomPCA** - plots all labelled instances in 2D using PCA for dimensional reduction.<br>
**04_plot_Isomap** - plots all labelled instances in 2D using Isomap for dimensional reduction.<br>
**05_multilabel** - classification that will return multi-label if applicable on tests.<br>
**06_k_fold_cross_validation** - 10-fold CV for all possible classifiers, with averaged metrics ready (accuracy, precision, recall, f1).<br>
**07_k_fold_cross_validation_multi_algorithms** - 10-fold CV for all possible classifiers, with averaged metrics ready (accuracy, precision, recall, f1), iterates through all classifiers in one run.<br>
**08_para_tunning** - tune the parameters for classifiers on the 717 annotated set.<br>
**09_save_classication_models** - used to save the models.<br>
**10_grading_classify** - grading coverage of a given privacy policy, based on the results from Ridge classifier.<br>
**11_distinguish_nonpp** - classifier that can distinguish if a paragraph is privacy policy paragraph or not -- currently, only one label is shown if a paragraph is PP.<br>
**12_grading_xml** - grading coverage of a given privacy policy, then export into a `standard' structure in xml for visualization.<br>

**13_em_gmm** - semi-supervised classification using EM.<br>
**14_feature_selection** - running tests on chi2 feature select (100,2500,100) for classifiers and then save the results in .csv files.<br>

**15_grading_naive** - implement the Chrome naive grader in Python.<br>

Warning: Below are just sample text for further usage.
================================

Oh, and one thing I cannot stand is the mangling of words with multiple underscores in them like perform_complicated_task or do_this_and_do_that_and_another_thing.

A bit of the GitHub spice
-------------------------

In addition to the changes in the previous section, certain references are auto-linked:

* SHA: be6a8cc1c1ecfe9489fb51e4869af15a13fc2cd2
* User@SHA ref: mojombo@be6a8cc1c1ecfe9489fb51e4869af15a13fc2cd2
* User/Project@SHA: mojombo/god@be6a8cc1c1ecfe9489fb51e4869af15a13fc2cd2
* \#Num: #1
* User/#Num: mojombo#1
* User/Project#Num: mojombo/god#1

These are dangerous goodies though, and we need to make sure email addresses don't get mangled:

My email addy is tom@github.com.

Math is hard, let's go shopping
-------------------------------

In first grade I learned that 5 > 3 and 2 < 7. Maybe some arrows. 1 -> 2 -> 3. 9 <- 8 <- 7.

Triangles man! a^2 + b^2 = c^2

We all like making lists
------------------------

The above header should be an H2 tag. Now, for a list of fruits:

* Red Apples
* Purple Grapes
* Green Kiwifruits

Let's get crazy:

1.  This is a list item with two paragraphs. Lorem ipsum dolor
    sit amet, consectetuer adipiscing elit. Aliquam hendrerit
    mi posuere lectus.

    Vestibulum enim wisi, viverra nec, fringilla in, laoreet
    vitae, risus. Donec sit amet nisl. Aliquam semper ipsum
    sit amet velit.

2.  Suspendisse id sem consectetuer libero luctus adipiscing.

What about some code **in** a list? That's insane, right?

1. In Ruby you can map like this:

        ['a', 'b'].map { |x| x.uppercase }

2. In Rails, you can do a shortcut:

        ['a', 'b'].map(&:uppercase)

Some people seem to like definition lists

<dl>
  <dt>Lower cost</dt>
  <dd>The new version of this product costs significantly less than the previous one!</dd>
  <dt>Easier to use</dt>
  <dd>We've changed the product so that it's much easier to use!</dd>
</dl>

I am a robot
------------

Maybe you want to print `robot` to the console 1000 times. Why not?

    def robot_invasion
      puts("robot " * 1000)
    end

You see, that was formatted as code because it's been indented by four spaces.

How about we throw some angle braces and ampersands in there?

    <div class="footer">
        &copy; 2004 Foo Corporation
    </div>

Set in stone
------------

Preformatted blocks are useful for ASCII art:

<pre>
             ,-. 
    ,     ,-.   ,-. 
   / \   (   )-(   ) 
   \ |  ,.>-(   )-< 
    \|,' (   )-(   ) 
     Y ___`-'   `-' 
     |/__/   `-' 
     | 
     | 
     |    -hrr- 
  ___|_____________ 
</pre>

Playing the blame game
----------------------

If you need to blame someone, the best way to do so is by quoting them:

> I, at any rate, am convinced that He does not throw dice.

Or perhaps someone a little less eloquent:

> I wish you'd have given me this written question ahead of time so I
> could plan for it... I'm sure something will pop into my head here in
> the midst of this press conference, with all the pressure of trying to
> come up with answer, but it hadn't yet...
>
> I don't want to sound like
> I have made no mistakes. I'm confident I have. I just haven't - you
> just put me under the spot here, and maybe I'm not as quick on my feet
> as I should be in coming up with one.

Table for two
-------------

<table>
  <tr>
    <th>ID</th><th>Name</th><th>Rank</th>
  </tr>
  <tr>
    <td>1</td><td>Tom Preston-Werner</td><td>Awesome</td>
  </tr>
  <tr>
    <td>2</td><td>Albert Einstein</td><td>Nearly as awesome</td>
  </tr>
</table>

Crazy linking action
--------------------

I get 10 times more traffic from [Google] [1] than from
[Yahoo] [2] or [MSN] [3].

  [1]: http://google.com/        "Google"
  [2]: http://search.yahoo.com/  "Yahoo Search"
  [3]: http://search.msn.com/    "MSN Search"