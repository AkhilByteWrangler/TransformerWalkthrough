import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Walkthrough } from './components/Walkthrough';
import './App.css';

const App: React.FC = () => {
  return (
    <div className="app">
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      >
        <h1>Transformer</h1>
        <p>Architecture Birds-Eye Walkthrough</p>
      </motion.header>

      <Walkthrough />

      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.5 }}
      >
        <p>Built for Duke University • Intelligent Agents Course</p>
      </motion.footer>
    </div>
  );
};

export default App;
