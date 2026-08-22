package com.sih.document_screening.model;

import java.util.UUID;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.Setter;

@Entity
@Getter
@Setter
@Table(name = "verification_scores")
public class VerificationScore {
     @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @OneToOne
    @JoinColumn(name = "screening_id", nullable = false)
    private Screening screening;

    private Double faceMatchScore;         // 0.0 - 1.0
    private Double photoTamperingScore;    // 0.0 - 1.0 (ELA / Splice detection)
    private Double textManipulationScore;  // 0.0 - 1.0
    private Boolean metadataAnomalyFound;
}
