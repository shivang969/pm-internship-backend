import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import logging
import uuid
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PM Internship Scheme API",
    description="AI-Powered Internship Matching with ML",
    version="3.0.0"
)

# CORS Configuration
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CandidateCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str
    skills: List[str]
    qualifications: List[str]
    location_preference: List[str]
    current_location: str
    category: str  # General, OBC, SC, ST
    district_type: str  # Urban, Rural, Aspirational
    past_participation: bool = False
    experience_months: int = 0
    preferred_sectors: List[str] = []
    languages: List[str] = ["English"]

class IndustryCreate(BaseModel):
    company_name: str
    contact_email: EmailStr
    contact_phone: str
    internship_title: str
    internship_description: str
    required_skills: List[str]
    preferred_qualifications: List[str]
    location: str
    sector: str
    internship_capacity: int
    duration_months: int
    stipend_range: str = "Not specified"
    remote_allowed: bool = False
    preferred_candidate_profile: str = ""

class MatchRequest(BaseModel):
    candidate_id: Optional[str] = None
    industry_id: Optional[str] = None
    top_n: int = 10
    min_score_threshold: float = 0.3
    use_ml_prediction: bool = True

class FeedbackCreate(BaseModel):
    candidate_id: str
    industry_id: str
    placement_successful: bool
    rating: Optional[float] = None
    feedback_text: Optional[str] = None

# ==================== IN-MEMORY STORAGE ====================

candidates_db: Dict[str, Dict[str, Any]] = {}
industries_db: Dict[str, Dict[str, Any]] = {}
feedback_db: List[Dict[str, Any]] = []
matches_db: List[Dict[str, Any]] = []

# ==================== ML ENGINE ====================

class MLMatchingEngine:
    """
    Enhanced ML Matching Engine using:
    - Linear Regression for score prediction
    - Logistic Regression for classification
    - K-Means for clustering
    - TF-IDF for text similarity
    """
    
    def __init__(self):
        self.linear_model = None
        self.logistic_model = None
        self.candidate_kmeans = None
        self.industry_kmeans = None
        self.scaler = StandardScaler()
        self.n_clusters = 3
        self._initialize_models()
    
    def _initialize_models(self):
        """Train ML models with synthetic data"""
        np.random.seed(42)
        n_samples = 200
        
        # Generate training features
        X_train = np.random.rand(n_samples, 6)
        
        # Linear Regression target (continuous scores)
        y_linear = (
            X_train[:, 0] * 0.40 +  # skills
            X_train[:, 1] * 0.20 +  # location
            X_train[:, 2] * 0.15 +  # qualification
            X_train[:, 3] * 0.10 +  # experience
            X_train[:, 4] * 0.10 +  # category bonus
            X_train[:, 5] * 0.05 +  # sector
            np.random.normal(0, 0.05, n_samples)
        )
        y_linear = np.clip(y_linear, 0, 1)
        
        # Logistic Regression target (binary classification)
        y_logistic = (y_linear > 0.6).astype(int)
        
        # Train models
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_linear)
        
        self.logistic_model = LogisticRegression(random_state=42, max_iter=200)
        self.logistic_model.fit(X_train, y_logistic)
        
        self.candidate_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.industry_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        
        logger.info(f"✓ ML Models Initialized - Linear R²: {self.linear_model.score(X_train, y_linear):.3f}")
        logger.info(f"✓ Logistic Accuracy: {self.logistic_model.score(X_train, y_logistic):.3f}")
    
    def extract_features(self, candidate: Dict, industry: Dict) -> np.ndarray:
        """Extract numerical features for ML prediction"""
        
        # 1. Skills similarity (TF-IDF)
        skills_score = self._tfidf_similarity(
            candidate.get("skills", []),
            industry.get("required_skills", [])
        )
        
        # 2. Location match
        location_score = self._location_score(
            candidate.get("location_preference", []),
            industry.get("location", ""),
            industry.get("remote_allowed", False)
        )
        
        # 3. Qualification match
        qual_score = self._qualification_score(
            candidate.get("qualifications", []),
            industry.get("preferred_qualifications", [])
        )
        
        # 4. Experience (normalized)
        exp_score = min(candidate.get("experience_months", 0) / 24.0, 1.0)
        
        # 5. Affirmative action bonus
        category_map = {"ST": 0.30, "SC": 0.25, "OBC": 0.18, "General": 0.0}
        district_map = {"Aspirational": 0.20, "Rural": 0.15, "Urban": 0.0}
        category_bonus = (
            category_map.get(candidate.get("category", "General"), 0.0) +
            district_map.get(candidate.get("district_type", "Urban"), 0.0)
        )
        category_bonus = min(category_bonus, 0.5)
        
        # 6. Sector match
        sector_score = 1.0 if industry.get("sector") in candidate.get("preferred_sectors", []) else 0.3
        
        return np.array([
            skills_score,
            location_score,
            qual_score,
            exp_score,
            category_bonus,
            sector_score
        ])
    
    def _tfidf_similarity(self, skills1: List[str], skills2: List[str]) -> float:
        """Calculate TF-IDF cosine similarity"""
        if not skills1 or not skills2:
            return 0.0
        
        doc1 = " ".join(s.lower() for s in skills1)
        doc2 = " ".join(s.lower() for s in skills2)
        
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([doc1, doc2])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(similarity)
        except:
            # Fallback: Jaccard similarity
            set1 = set(s.lower() for s in skills1)
            set2 = set(s.lower() for s in skills2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
    
    def _location_score(self, prefs: List[str], location: str, remote: bool) -> float:
        """Calculate location match score"""
        if remote:
            return 1.0
        if not prefs:
            return 0.5
        
        location_lower = location.lower()
        for pref in prefs:
            if pref.lower() == location_lower:
                return 1.0
        
        # Regional proximity
        regions = {
            "delhi": ["ncr", "noida", "gurgaon"],
            "mumbai": ["pune", "thane"],
            "bangalore": ["mysore"],
            "hyderabad": ["secunderabad"]
        }
        
        for pref in prefs:
            pref_lower = pref.lower()
            if pref_lower in regions.get(location_lower, []):
                return 0.7
            if location_lower in regions.get(pref_lower, []):
                return 0.7
        
        return 0.2
    
    def _qualification_score(self, cand_quals: List[str], pref_quals: List[str]) -> float:
        """Calculate qualification match"""
        if not pref_quals:
            return 0.6
        
        cand_set = set(q.lower() for q in cand_quals)
        pref_set = set(q.lower() for q in pref_quals)
        
        if cand_set & pref_set:
            return 1.0
        
        # Partial keyword match
        for cq in cand_set:
            for pq in pref_set:
                if any(word in cq for word in pq.split()) or any(word in pq for word in cq.split()):
                    return 0.7
        
        return 0.3
    
    def predict_match(self, candidate: Dict, industry: Dict, use_ml: bool = True) -> Dict[str, Any]:
        """Generate match score with ML predictions"""
        
        features = self.extract_features(candidate, industry)
        
        if use_ml and self.linear_model and self.logistic_model:
            # Linear Regression prediction
            linear_score = self.linear_model.predict(features.reshape(1, -1))[0]
            linear_score = float(np.clip(linear_score, 0, 1))
            
            # Logistic Regression prediction
            logistic_class = self.logistic_model.predict(features.reshape(1, -1))[0]
            logistic_prob = self.logistic_model.predict_proba(features.reshape(1, -1))[0][1]
            
            # Combined ML score
            ml_score = (linear_score * 0.6 + logistic_prob * 0.4)
            category_bonus = features[4] * 0.5
            final_score = min(ml_score + category_bonus, 1.0)
            
            return {
                "overall_score": round(final_score, 3),
                "ml_linear_prediction": round(linear_score, 3),
                "ml_logistic_probability": round(logistic_prob, 3),
                "ml_logistic_class": "Good Match" if logistic_class == 1 else "Poor Match",
                "skills_score": round(features[0], 3),
                "location_score": round(features[1], 3),
                "qualification_score": round(features[2], 3),
                "experience_score": round(features[3], 3),
                "affirmative_bonus": round(features[4], 3),
                "sector_score": round(features[5], 3),
                "ml_enabled": True
            }
        else:
            # Rule-based fallback
            base_score = (
                features[0] * 0.35 +
                features[1] * 0.20 +
                features[2] * 0.15 +
                features[5] * 0.15 +
                features[3] * 0.05 +
                0.10
            )
            final_score = min(base_score + features[4] * 0.5, 1.0)
            
            return {
                "overall_score": round(final_score, 3),
                "skills_score": round(features[0], 3),
                "location_score": round(features[1], 3),
                "qualification_score": round(features[2], 3),
                "experience_score": round(features[3], 3),
                "affirmative_bonus": round(features[4], 3),
                "sector_score": round(features[5], 3),
                "ml_enabled": False
            }
    
    def cluster_candidates(self, candidates: Dict[str, Dict]) -> Dict[str, Any]:
        """Apply K-Means clustering to candidates"""
        if len(candidates) < self.n_clusters:
            return {"error": "Not enough candidates for clustering"}
        
        feature_matrix = []
        candidate_ids = []
        
        dummy_industry = {
            "required_skills": [],
            "location": "",
            "remote_allowed": True,
            "preferred_qualifications": [],
            "sector": ""
        }
        
        for cid, candidate in candidates.items():
            features = self.extract_features(candidate, dummy_industry)
            feature_matrix.append(features)
            candidate_ids.append(cid)
        
        X = np.array(feature_matrix)
        self.candidate_kmeans.fit(X)
        labels = self.candidate_kmeans.labels_
        
        clusters = {}
        for idx, label in enumerate(labels):
            cluster_name = f"Cluster_{label}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append({
                "candidate_id": candidate_ids[idx],
                "name": candidates[candidate_ids[idx]].get("name")
            })
        
        return {
            "n_clusters": self.n_clusters,
            "clusters": clusters,
            "total_candidates": len(candidates)
        }

# Initialize ML Engine
ml_engine = MLMatchingEngine()

# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 PM Internship Scheme API Started")
    logger.info("✓ ML Models: Linear Regression, Logistic Regression, K-Means")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PM Internship Scheme - AI-Powered Matching API",
        "version": "3.0.0",
        "status": "active",
        "ml_models": ["Linear Regression", "Logistic Regression", "K-Means", "TF-IDF"],
        "endpoints": {
            "candidates": "/candidates",
            "industries": "/industries",
            "match": "/match_internships",
            "stats": "/stats",
            "clustering": "/cluster_candidates",
            "ml_info": "/ml_info"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/candidates")
async def create_candidate(candidate: CandidateCreate):
    """Register a new candidate"""
    try:
        candidate_id = str(uuid.uuid4())
        candidate_data = candidate.dict()
        candidate_data.update({
            "id": candidate_id,
            "registration_date": datetime.now().isoformat(),
            "status": "active"
        })
        
        candidates_db[candidate_id] = candidate_data
        logger.info(f"✓ Candidate registered: {candidate.name} ({candidate_id})")
        
        return {
            "status": "success",
            "message": "Candidate registered successfully",
            "candidate_id": candidate_id,
            "data": candidate_data
        }
    except Exception as e:
        logger.error(f"Error registering candidate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates")
async def get_candidates():
    """Get all candidates"""
    return {
        "status": "success",
        "count": len(candidates_db),
        "candidates": list(candidates_db.values())
    }

@app.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str):
    """Get specific candidate"""
    if candidate_id not in candidates_db:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return {
        "status": "success",
        "candidate": candidates_db[candidate_id]
    }

@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str):
    """Delete a candidate"""
    if candidate_id not in candidates_db:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    deleted = candidates_db.pop(candidate_id)
    return {
        "status": "success",
        "message": "Candidate deleted",
        "deleted_candidate": deleted
    }

@app.post("/industries")
async def create_industry(industry: IndustryCreate):
    """Register a new industry"""
    try:
        industry_id = str(uuid.uuid4())
        industry_data = industry.dict()
        industry_data.update({
            "id": industry_id,
            "registration_date": datetime.now().isoformat(),
            "status": "active",
            "filled_positions": 0
        })
        
        industries_db[industry_id] = industry_data
        logger.info(f"✓ Industry registered: {industry.company_name} ({industry_id})")
        
        return {
            "status": "success",
            "message": "Industry registered successfully",
            "industry_id": industry_id,
            "data": industry_data
        }
    except Exception as e:
        logger.error(f"Error registering industry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/industries")
async def get_industries():
    """Get all industries"""
    return {
        "status": "success",
        "count": len(industries_db),
        "industries": list(industries_db.values())
    }

@app.get("/industries/{industry_id}")
async def get_industry(industry_id: str):
    """Get specific industry"""
    if industry_id not in industries_db:
        raise HTTPException(status_code=404, detail="Industry not found")
    
    return {
        "status": "success",
        "industry": industries_db[industry_id]
    }

@app.delete("/industries/{industry_id}")
async def delete_industry(industry_id: str):
    """Delete an industry"""
    if industry_id not in industries_db:
        raise HTTPException(status_code=404, detail="Industry not found")
    
    deleted = industries_db.pop(industry_id)
    return {
        "status": "success",
        "message": "Industry deleted",
        "deleted_industry": deleted
    }

@app.post("/match_internships")
async def match_internships(request: MatchRequest):
    """AI-powered internship matching"""
    try:
        matches = []
        
        # Match specific candidate with all industries
        if request.candidate_id:
            if request.candidate_id not in candidates_db:
                raise HTTPException(status_code=404, detail="Candidate not found")
            
            candidate = candidates_db[request.candidate_id]
            
            for industry_id, industry in industries_db.items():
                if industry.get("filled_positions", 0) < industry.get("internship_capacity", 0):
                    score = ml_engine.predict_match(candidate, industry, request.use_ml_prediction)
                    
                    if score["overall_score"] >= request.min_score_threshold:
                        matches.append({
                            "candidate_id": request.candidate_id,
                            "candidate_name": candidate.get("name"),
                            "industry_id": industry_id,
                            "company_name": industry.get("company_name"),
                            "internship_title": industry.get("internship_title"),
                            "match_score": score,
                            "available_positions": industry.get("internship_capacity", 0) - industry.get("filled_positions", 0)
                        })
        
        # Match specific industry with all candidates
        elif request.industry_id:
            if request.industry_id not in industries_db:
                raise HTTPException(status_code=404, detail="Industry not found")
            
            industry = industries_db[request.industry_id]
            
            for candidate_id, candidate in candidates_db.items():
                score = ml_engine.predict_match(candidate, industry, request.use_ml_prediction)
                
                if score["overall_score"] >= request.min_score_threshold:
                    matches.append({
                        "candidate_id": candidate_id,
                        "candidate_name": candidate.get("name"),
                        "industry_id": request.industry_id,
                        "company_name": industry.get("company_name"),
                        "internship_title": industry.get("internship_title"),
                        "match_score": score,
                        "available_positions": industry.get("internship_capacity", 0) - industry.get("filled_positions", 0)
                    })
        
        # Match all candidates with all industries
        else:
            for candidate_id, candidate in candidates_db.items():
                for industry_id, industry in industries_db.items():
                    if industry.get("filled_positions", 0) < industry.get("internship_capacity", 0):
                        score = ml_engine.predict_match(candidate, industry, request.use_ml_prediction)
                        
                        if score["overall_score"] >= request.min_score_threshold:
                            matches.append({
                                "candidate_id": candidate_id,
                                "candidate_name": candidate.get("name"),
                                "industry_id": industry_id,
                                "company_name": industry.get("company_name"),
                                "internship_title": industry.get("internship_title"),
                                "match_score": score,
                                "available_positions": industry.get("internship_capacity", 0) - industry.get("filled_positions", 0)
                            })
        
        # Sort by score and limit
        matches.sort(key=lambda x: x["match_score"]["overall_score"], reverse=True)
        matches = matches[:request.top_n]
        
        # Store matches
        for match in matches:
            match_entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                **match
            }
            matches_db.append(match_entry)
        
        logger.info(f"✓ Generated {len(matches)} matches")
        
        return {
            "status": "success",
            "total_matches": len(matches),
            "matches": matches,
            "ml_enabled": request.use_ml_prediction,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Matching error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    
    # Category distribution
    category_dist = {}
    district_dist = {}
    for candidate in candidates_db.values():
        cat = candidate.get("category", "Unknown")
        dist = candidate.get("district_type", "Unknown")
        category_dist[cat] = category_dist.get(cat, 0) + 1
        district_dist[dist] = district_dist.get(dist, 0) + 1
    
    # Sector distribution
    sector_dist = {}
    for industry in industries_db.values():
        sector = industry.get("sector", "Unknown")
        sector_dist[sector] = sector_dist.get(sector, 0) + 1
    
    # Capacity stats
    total_capacity = sum(i.get("internship_capacity", 0) for i in industries_db.values())
    filled_positions = sum(i.get("filled_positions", 0) for i in industries_db.values())
    
    return {
        "status": "success",
        "candidates": {
            "total": len(candidates_db),
            "active": sum(1 for c in candidates_db.values() if c.get("status") == "active"),
            "by_category": category_dist,
            "by_district": district_dist
        },
        "industries": {
            "total": len(industries_db),
            "active": sum(1 for i in industries_db.values() if i.get("status") == "active"),
            "by_sector": sector_dist
        },
        "internships": {
            "total_capacity": total_capacity,
            "filled_positions": filled_positions,
            "available_positions": total_capacity - filled_positions,
            "utilization_rate": round((filled_positions / total_capacity * 100), 2) if total_capacity > 0 else 0
        },
        "matches": {
            "total_generated": len(matches_db)
        },
        "feedback": {
            "total_received": len(feedback_db)
        },
        "ml_models_active": 3,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cluster_candidates")
async def cluster_candidates():
    """Apply K-Means clustering to candidates"""
    try:
        result = ml_engine.cluster_candidates(candidates_db)
        return {
            "status": "success",
            "clustering_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml_info")
async def get_ml_info():
    """Get ML model information"""
    return {
        "status": "success",
        "ml_models": {
            "linear_regression": {
                "purpose": "Predicts continuous match score (0 to 1)",
                "algorithm": "Ordinary Least Squares",
                "features": [
                    "skills_match",
                    "location_match",
                    "qualification_match",
                    "experience_normalized",
                    "affirmative_bonus",
                    "sector_match"
                ],
                "trained": True,
                "coefficients": ml_engine.linear_model.coef_.tolist() if ml_engine.linear_model else None
            },
            "logistic_regression": {
                "purpose": "Binary classification (Good Match / Poor Match)",
                "algorithm": "Logistic Regression with L2 regularization",
                "features": "Same as Linear Regression",
                "trained": True,
                "classes": ["Poor Match (0)", "Good Match (1)"]
            },
            "kmeans_clustering": {
                "purpose": "Groups similar candidates into clusters",
                "algorithm": "K-Means Clustering",
                "n_clusters": ml_engine.n_clusters,
                "trained": True
            },
            "tfidf": {
                "purpose": "Text similarity for skills matching",
                "algorithm": "TF-IDF Vectorization + Cosine Similarity",
                "status": "Active"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackCreate):
    """Submit feedback for model improvement"""
    try:
        if feedback.candidate_id not in candidates_db:
            raise HTTPException(status_code=404, detail="Candidate not found")
        if feedback.industry_id not in industries_db:
            raise HTTPException(status_code=404, detail="Industry not found")
        
        feedback_entry = feedback.dict()
        feedback_entry.update({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        })
        
        feedback_db.append(feedback_entry)
        
        logger.info(f"✓ Feedback received: {feedback.candidate_id} → {feedback.industry_id}")
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "feedback_id": feedback_entry["id"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback")
async def get_feedback():
    """Get all feedback"""
    return {
        "status": "success",
        "count": len(feedback_db),
        "feedback": feedback_db
    }

@app.delete("/reset")
async def reset_system():
    """Reset all data (use with caution)"""
    global candidates_db, industries_db, feedback_db, matches_db
    
    candidates_db.clear()
    industries_db.clear()
    feedback_db.clear()
    matches_db.clear()
    
    logger.warning("⚠️  System reset - all data cleared")
    
    return {
        "status": "success",
        "message": "System reset successfully",
        "timestamp": datetime.now().isoformat()
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_version": "3.0.0",
        "ml_models": "active",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")