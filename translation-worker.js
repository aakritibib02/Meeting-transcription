/**
 * Translation Worker - Handles translation tasks using free services
 */

class TranslationWorker {
    constructor() {
        this.cache = new Map(); // Simple translation cache
        this.rateLimitQueue = [];
        this.isProcessingQueue = false;
        this.lastRequestTime = 0;
        this.minRequestInterval = 1000; // 1 second between requests to avoid rate limiting
    }

    /**
     * Translate text using free translation services
     * @param {string} text - Text to translate
     * @param {string} targetLang - Target language code
     * @param {string} sourceLang - Source language code (optional)
     * @returns {Promise<string>} - Translated text
     */
    async translate(text, targetLang, sourceLang = 'auto') {
        if (!text || !text.trim()) {
            return '';
        }

        // Check cache first
        const cacheKey = `${sourceLang}-${targetLang}-${text}`;
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            // Try multiple translation methods in order of preference
            let translation = null;

            // Method 1: Use browser's built-in translation API if available
            translation = await this.tryBrowserTranslation(text, targetLang, sourceLang);
            
            if (!translation) {
                // Method 2: Use LibreTranslate (free, open-source)
                translation = await this.tryLibreTranslate(text, targetLang, sourceLang);
            }

            if (!translation) {
                // Method 3: Use MyMemory (free tier)
                translation = await this.tryMyMemoryTranslation(text, targetLang, sourceLang);
            }

            if (!translation) {
                // Method 4: Simple word substitution for common phrases (fallback)
                translation = this.trySimpleTranslation(text, targetLang);
            }

            // Cache the result
            if (translation) {
                this.cache.set(cacheKey, translation);
                
                // Limit cache size
                if (this.cache.size > 1000) {
                    const firstKey = this.cache.keys().next().value;
                    this.cache.delete(firstKey);
                }
            }

            return translation || text; // Return original if no translation found

        } catch (error) {
            console.error('Translation error:', error);
            return text; // Return original text on error
        }
    }

    /**
     * Try using browser's built-in translation capabilities
     */
    async tryBrowserTranslation(text, targetLang, sourceLang) {
        try {
            // This would use the browser's translation API if available
            // Currently, there's no standard browser translation API
            // This is a placeholder for future implementation
            
            if ('translation' in window && window.translation) {
                const result = await window.translation.translate({
                    text: text,
                    to: targetLang,
                    from: sourceLang
                });
                return result.text;
            }
            
            return null;
        } catch (error) {
            console.error('Browser translation failed:', error);
            return null;
        }
    }

    /**
     * Try LibreTranslate (free, open-source translation service)
     */
    async tryLibreTranslate(text, targetLang, sourceLang) {
        try {
            // LibreTranslate free API endpoints (you might need to find a public instance)
            const endpoints = [
                'https://libretranslate.de/translate',
                'https://libretranslate.com/translate'
            ];

            for (const endpoint of endpoints) {
                try {
                    const response = await this.makeRateLimitedRequest(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            q: text,
                            source: sourceLang === 'auto' ? 'auto' : this.mapLanguageCode(sourceLang, 'libretranslate'),
                            target: this.mapLanguageCode(targetLang, 'libretranslate'),
                            format: 'text'
                        })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        return data.translatedText;
                    }
                } catch (error) {
                    console.warn(`LibreTranslate endpoint ${endpoint} failed:`, error);
                    continue;
                }
            }
            
            return null;
        } catch (error) {
            console.error('LibreTranslate failed:', error);
            return null;
        }
    }

    /**
     * Try MyMemory translation service (free tier available)
     */
    async tryMyMemoryTranslation(text, targetLang, sourceLang) {
        try {
            const langPair = sourceLang === 'auto' ? `en|${targetLang}` : `${sourceLang}|${targetLang}`;
            const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${langPair}`;

            const response = await this.makeRateLimitedRequest(url);
            
            if (response.ok) {
                const data = await response.json();
                if (data.responseStatus === 200) {
                    return data.responseData.translatedText;
                }
            }
            
            return null;
        } catch (error) {
            console.error('MyMemory translation failed:', error);
            return null;
        }
    }

    /**
     * Simple word/phrase substitution for common terms (fallback method)
     */
    trySimpleTranslation(text, targetLang) {
        const commonPhrases = {
            'es': {
                'hello': 'hola',
                'goodbye': 'adiós',
                'thank you': 'gracias',
                'please': 'por favor',
                'yes': 'sí',
                'no': 'no',
                'meeting': 'reunión',
                'task': 'tarea',
                'action item': 'elemento de acción',
                'deadline': 'fecha límite',
                'urgent': 'urgente',
                'important': 'importante'
            },
            'fr': {
                'hello': 'bonjour',
                'goodbye': 'au revoir',
                'thank you': 'merci',
                'please': 's\'il vous plaît',
                'yes': 'oui',
                'no': 'non',
                'meeting': 'réunion',
                'task': 'tâche',
                'action item': 'élément d\'action',
                'deadline': 'échéance',
                'urgent': 'urgent',
                'important': 'important'
            },
            'de': {
                'hello': 'hallo',
                'goodbye': 'auf wiedersehen',
                'thank you': 'danke',
                'please': 'bitte',
                'yes': 'ja',
                'no': 'nein',
                'meeting': 'besprechung',
                'task': 'aufgabe',
                'action item': 'aktionspunkt',
                'deadline': 'frist',
                'urgent': 'dringend',
                'important': 'wichtig'
            }
        };

        const translations = commonPhrases[targetLang];
        if (!translations) {
            return null;
        }

        let translatedText = text.toLowerCase();
        
        for (const [english, translated] of Object.entries(translations)) {
            const regex = new RegExp(`\\b${english}\\b`, 'gi');
            translatedText = translatedText.replace(regex, translated);
        }

        return translatedText !== text.toLowerCase() ? translatedText : null;
    }

    /**
     * Make a rate-limited request to avoid hitting API limits
     */
    async makeRateLimitedRequest(url, options = {}) {
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        
        if (timeSinceLastRequest < this.minRequestInterval) {
            await new Promise(resolve => setTimeout(resolve, this.minRequestInterval - timeSinceLastRequest));
        }

        this.lastRequestTime = Date.now();
        
        // Set a reasonable timeout
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000); // 10 second timeout

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(timeout);
            return response;
        } catch (error) {
            clearTimeout(timeout);
            throw error;
        }
    }

    /**
     * Map language codes between different services
     */
    mapLanguageCode(langCode, service) {
        const mappings = {
            'libretranslate': {
                'en-US': 'en',
                'en-GB': 'en',
                'es-ES': 'es',
                'fr-FR': 'fr',
                'de-DE': 'de',
                'it-IT': 'it',
                'pt-BR': 'pt',
                'zh-CN': 'zh',
                'ja-JP': 'ja',
                'ko-KR': 'ko'
            }
        };

        const serviceMapping = mappings[service];
        return serviceMapping ? (serviceMapping[langCode] || langCode) : langCode;
    }

    /**
     * Detect language of text (simple heuristic-based approach)
     */
    detectLanguage(text) {
        if (!text || text.length < 10) {
            return 'en'; // Default to English
        }

        // Simple keyword-based language detection
        const patterns = {
            'es': /\b(el|la|los|las|de|que|y|es|en|un|una|con|se|no|te|lo|le|da|su|por|son|como|para|pero|muy|todo|ser|ya|estar|tener|hacer|poder|decir|ir|ver|saber|dar|querer|venir|tiempo|casa|vida|día|hombre|mundo|año|trabajo|parte|niño)\b/gi,
            'fr': /\b(le|de|et|à|un|il|être|et|en|avoir|que|pour|dans|ce|son|une|sur|avec|ne|se|pas|tout|pouvoir|par|plus|aller|faire|savoir|être|grand|nouveau|partir|temps|prendre|vie|jour|homme|enfant|an|travail|famille|dire|maison)\b/gi,
            'de': /\b(der|die|und|in|den|von|zu|das|mit|sich|des|auf|für|ist|im|dem|nicht|ein|eine|als|auch|es|an|werden|aus|er|hat|dass|sie|nach|wird|bei|noch|wie|einem|über|einen|so|zum|war|haben|nur|oder|aber|vor|zur|bis|unter|zwei)\b/gi,
            'it': /\b(il|di|che|e|la|per|in|un|è|non|da|a|con|del|le|si|più|su|una|come|essere|questo|tutto|anche|molto|bene|fare|avere|dire|dovere|altro|grande|stesso|proprio|vedere|sapere|dare|stare|volere|casa|tempo|anno|giorno|vita)\b/gi
        };

        let bestMatch = 'en';
        let bestScore = 0;

        for (const [lang, pattern] of Object.entries(patterns)) {
            const matches = (text.match(pattern) || []).length;
            const score = matches / text.split(' ').length;
            
            if (score > bestScore) {
                bestScore = score;
                bestMatch = lang;
            }
        }

        return bestScore > 0.1 ? bestMatch : 'en';
    }

    /**
     * Clear translation cache
     */
    clearCache() {
        this.cache.clear();
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TranslationWorker;
} else if (typeof window !== 'undefined') {
    window.TranslationWorker = TranslationWorker;
}

// Create a global instance for the extension
if (typeof chrome !== 'undefined') {
    const translationWorker = new TranslationWorker();
    
    // Make it available globally
    if (typeof window !== 'undefined') {
        window.translationWorker = translationWorker;
    }
}