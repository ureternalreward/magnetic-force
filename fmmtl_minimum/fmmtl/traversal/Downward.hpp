#pragma once

/** @brief Process the boxes from top to bottom
 * concept Tree {
 *   unsigned levels();                       // Levels in the tree, root:0
 *   box_iterator box_begin(unsigned level);  // Iterator range to boxes of level
 *   box_iterator box_end(unsigned level);
 * }
 * concept Evaluator {
 *   void operator()(typename Tree::box_type& b);   // Process box b
 * }
 */
struct DownwardPass {
  template <class Tree, class Evaluator>
  inline static void eval(Tree& tree, Evaluator& eval) {
    // For the highest level down to the lowest level
    for (unsigned l = 0; l < tree.levels(); ++l) {
      // For all boxes at this level
      auto b_end = tree.box_end(l);
      std::vector<decltype(tree.box_begin(l))> level_box_list;
      level_box_list.reserve(tree.boxes(l));
      for (auto bit = tree.box_begin(l); bit != b_end; ++bit) {
        level_box_list.push_back(bit);
      }
#pragma omp parallel for
      for (int i = 0; i < level_box_list.size();i++) {
        auto box = *level_box_list[i];
        eval(box);
      }
      
    }
  }
};
